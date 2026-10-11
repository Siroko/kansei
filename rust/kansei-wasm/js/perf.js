// kansei_wasm's performance overlay (src/perf.rs), after react-three-fiber's r3f-perf: a corner box
// with frames per second, the frame's CPU and GPU time and their graphs, draw calls, triangles,
// pipelines and GPU memory; click it for each pass's GPU time, the CPU sections and the counts.
//
// - `?perf=1` shows it, the backquote key (`) toggles it, and so do perfSetVisible and
//   window.kanseiPerf ({ show, hide, toggle, visible, report }).
// - Rust's frame loop (kansei_wasm::run) calls perfFrame(cpuMs) once a frame while it shows;
//   kansei_wasm::perf::profile hands it the renderer's profile (perfProfile(json)).
// - It counts what the page asks of WebGPU: buffers and textures as they are made and freed
//   (hooked when this module loads, before the device exists, so the memory is all of it; a
//   texture's size from its format, mips and samples: an estimate; freed by destroy() or, dropped
//   undestroyed as wgpu's web backend does, by the collector), and while it shows, the draws and
//   dispatches the passes encode (their hooks come off when it hides), each by pass (the render
//   pass's label) and by material (group 0's bind group's label), render bundles counted where
//   they are executed. An indirect draw's triangles are the GPU's to know: the cluster LOD cuts'
//   come from the renderer's cull readbacks (perfClusters), the other indirect draws' are not
//   counted; the rest are the direct draws' (indices or vertices over 3, times the instances).
// - The expanded view: the passes' GPU time, the CPU sections, the cluster cuts, draws and
//   triangles by pass and by material, the counts, memory by label and pipelines by material;
//   window.kanseiPerf.report() gives the same as data.
// - In a worker (no document) it does nothing.

const hasDom = typeof document !== 'undefined';
const query = hasDom ? new URLSearchParams(location.search) : null;
let visible = !!query && (query.get('perf') === '1' || query.get('perf') === 'true');

// ---- WebGPU counts -------------------------------------------------------------------------

// (collected: bytes freed by the garbage collector, not destroy(): resources the page dropped)
const memory = { buffers: 0, bufferBytes: 0, textures: 0, textureBytes: 0, collected: 0, renderPipelines: 0, computePipelines: 0, shaders: 0, bindGroups: 0 };
const sizes = new WeakMap();
// this frame's encoding, while the overlay shows
const frame = { draws: 0, indirect: 0, triangles: 0, dispatches: 0, renderPasses: 0, computePasses: 0 };

// What each pass and each material draws, this frame: pass label -> { passes, draws, indirect,
// triangles }, and material (group 0's bind group's label up to the last '/', or else the
// pipeline's) -> the same. A render
// bundle's draws are counted where it is executed: what it holds is noted when it is recorded
// (bundles are recorded seldom, so that hook stays on).
const byPass = new Map();
const byMaterial = new Map();
// an encoder's pipeline and group 0's bind group (a material's, which names it best: materials
// of one program share their pipelines), as set last
const stateOf = new WeakMap();
const bundleOf = new WeakMap();
const owner = (object) => {
    const label = (object && object.label) || '';
    const cut = label.lastIndexOf('/');
    return cut > 0 ? label.slice(0, cut) : label;
};
const material = (state) => (state && (owner(state.group) || owner(state.pipeline))) || '(unlabelled)';
const setState = (encoder, key, value) => {
    let state = stateOf.get(encoder);
    if (!state) stateOf.set(encoder, (state = { pipeline: null, group: null }));
    state[key] = value;
};
const bucket = (map, key) => {
    let b = map.get(key);
    if (!b) map.set(key, (b = { passes: 0, draws: 0, indirect: 0, triangles: 0 }));
    return b;
};

// bytes a texel (a block for compressed formats: their 4x4 block over 16 texels)
function texelBytes(format) {
    if (/^(r8|stencil8)/.test(format)) return 1;
    if (/^(rg8|r16|depth16)/.test(format)) return 2;
    if (/^(rgba8|bgra8|rg16|r32|rgb10a2|rg11b10|rgb9e5|depth24plus|depth32float$)/.test(format)) return 4;
    if (/^(rgba16|rg32|depth32float-stencil8)/.test(format)) return 8;
    if (/^rgba32/.test(format)) return 16;
    if (/^(bc1|bc4|etc2-rgb8|etc2-rgb8a1|eac-r11)/.test(format)) return 0.5;
    if (/^(bc|etc2|eac|astc-4x4)/.test(format)) return 1;
    if (/^astc-(\d+)x(\d+)/.test(format)) {
        const [, w, h] = format.match(/^astc-(\d+)x(\d+)/);
        return 16 / (w * h);
    }
    return 4;
}

function textureBytes(desc) {
    const size = Array.isArray(desc.size) ? { width: desc.size[0], height: desc.size[1] ?? 1, depthOrArrayLayers: desc.size[2] ?? 1 } : desc.size;
    let w = size.width;
    let h = size.height ?? 1;
    let d = size.depthOrArrayLayers ?? 1;
    const is3d = desc.dimension === '3d';
    let bytes = 0;
    for (let level = 0; level < (desc.mipLevelCount ?? 1); level++) {
        bytes += w * h * d * texelBytes(desc.format);
        w = Math.max(1, w >> 1);
        h = Math.max(1, h >> 1);
        if (is3d) d = Math.max(1, d >> 1);
    }
    return bytes * (desc.sampleCount ?? 1);
}

function wrap(proto, name, after) {
    const original = proto && proto[name];
    if (typeof original !== 'function') return null;
    proto[name] = function (...args) {
        const result = original.apply(this, args);
        after(result, args, this);
        return result;
    };
    return () => { proto[name] = original; };
}

// what holds the memory (alive, by label) and who made the pipelines (by material): label ->
// { count, bytes }
const memoryByLabel = new Map();
const pipelinesByMaterial = new Map();
// pipelines made since the overlay last refreshed (a page making them every frame shows here)
let pipelinesMade = 0;
const add = (map, key, n, bytes) => {
    const e = map.get(key) || { count: 0, bytes: 0 };
    e.count += n;
    e.bytes += bytes;
    if (e.count > 0) map.set(key, e);
    else map.delete(key);
};

if (hasDom && typeof GPUDevice !== 'undefined') {
    // freed by destroy() or, dropped without it (wgpu's web backend never destroys on drop), when
    // the page lets go of it and the collector takes it: the browser frees its memory then
    const free = (entry) => {
        if (entry.freed) return;
        entry.freed = true;
        memory[entry.kind]--;
        memory[entry.bytes] -= entry.size;
        add(memoryByLabel, entry.label, -1, -entry.size);
        if (entry.dropped) memory.collected += entry.size;
    };
    const collected = typeof FinalizationRegistry !== 'undefined' ? new FinalizationRegistry((entry) => { entry.dropped = true; free(entry); }) : null;
    const track = (object, kind, bytes, size, label) => {
        const entry = { kind, bytes, size, label, freed: false, dropped: false };
        sizes.set(object, entry);
        if (collected) collected.register(object, entry, object);
        memory[kind]++;
        memory[bytes] += size;
        add(memoryByLabel, label, 1, size);
    };
    const released = (_, __, object) => {
        const entry = sizes.get(object);
        if (entry === undefined) return;
        sizes.delete(object);
        if (collected) collected.unregister(object);
        free(entry);
    };
    wrap(GPUDevice.prototype, 'createBuffer', (buffer, [desc]) => track(buffer, 'buffers', 'bufferBytes', desc.size, desc.label || '(unlabelled buffer)'));
    wrap(GPUDevice.prototype, 'createTexture', (texture, [desc]) => track(texture, 'textures', 'textureBytes', textureBytes(desc), desc.label || '(unlabelled texture)'));
    wrap(GPUBuffer.prototype, 'destroy', released);
    wrap(GPUTexture.prototype, 'destroy', released);
    const made = (kind) => (pipeline, [desc]) => {
        memory[kind]++;
        pipelinesMade++;
        add(pipelinesByMaterial, owner(desc) || '(unlabelled)', 1, 0);
    };
    for (const name of ['createRenderPipeline', 'createRenderPipelineAsync']) wrap(GPUDevice.prototype, name, made('renderPipelines'));
    for (const name of ['createComputePipeline', 'createComputePipelineAsync']) wrap(GPUDevice.prototype, name, made('computePipelines'));
    wrap(GPUDevice.prototype, 'createShaderModule', () => memory.shaders++);
    wrap(GPUDevice.prototype, 'createBindGroup', () => memory.bindGroups++);
}

function tally(pass, name, draws, indirect, triangles) {
    frame.draws += draws;
    frame.indirect += indirect;
    frame.triangles += triangles;
    for (const b of [bucket(byPass, (pass && pass.label) || '(unlabelled pass)'), bucket(byMaterial, name)]) {
        b.draws += draws;
        b.indirect += indirect;
        b.triangles += triangles;
    }
}

if (hasDom && typeof GPURenderBundleEncoder !== 'undefined') {
    const proto = GPURenderBundleEncoder.prototype;
    const held = (encoder) => {
        let list = bundleOf.get(encoder);
        if (!list) bundleOf.set(encoder, (list = new Map()));
        return list;
    };
    const note = (encoder, indirect, triangles) => {
        const b = bucket(held(encoder), material(stateOf.get(encoder)));
        b.draws++;
        b.indirect += indirect;
        b.triangles += triangles;
    };
    wrap(proto, 'setPipeline', (_, [pipeline], encoder) => setState(encoder, 'pipeline', pipeline));
    wrap(proto, 'setBindGroup', (_, [index, group], encoder) => index === 0 && setState(encoder, 'group', group));
    wrap(proto, 'draw', (_, [vertices, instances = 1], encoder) => note(encoder, 0, Math.floor(vertices / 3) * instances));
    wrap(proto, 'drawIndexed', (_, [indices, instances = 1], encoder) => note(encoder, 0, Math.floor(indices / 3) * instances));
    wrap(proto, 'drawIndirect', (_, __, encoder) => note(encoder, 1, 0));
    wrap(proto, 'drawIndexedIndirect', (_, __, encoder) => note(encoder, 1, 0));
    wrap(proto, 'finish', (bundle, _, encoder) => bundleOf.set(bundle, bundleOf.get(encoder) || new Map()));
}

// the per-frame hooks, on while the overlay shows
let unhook = [];
function hookDraws(on) {
    unhook.forEach((undo) => undo && undo());
    unhook = [];
    if (!on || typeof GPURenderPassEncoder === 'undefined') return;
    const proto = GPURenderPassEncoder.prototype;
    const name = (pass) => material(stateOf.get(pass));
    unhook.push(wrap(proto, 'setPipeline', (_, [pipeline], pass) => setState(pass, 'pipeline', pipeline)));
    unhook.push(wrap(proto, 'setBindGroup', (_, [index, group], pass) => index === 0 && setState(pass, 'group', group)));
    unhook.push(wrap(proto, 'draw', (_, [vertices, instances = 1], pass) => tally(pass, name(pass), 1, 0, Math.floor(vertices / 3) * instances)));
    unhook.push(wrap(proto, 'drawIndexed', (_, [indices, instances = 1], pass) => tally(pass, name(pass), 1, 0, Math.floor(indices / 3) * instances)));
    unhook.push(wrap(proto, 'drawIndirect', (_, __, pass) => tally(pass, name(pass), 1, 1, 0)));
    unhook.push(wrap(proto, 'drawIndexedIndirect', (_, __, pass) => tally(pass, name(pass), 1, 1, 0)));
    unhook.push(wrap(proto, 'executeBundles', (_, [bundles], pass) => {
        for (const bundle of bundles) {
            for (const [label, b] of bundleOf.get(bundle) || []) tally(pass, label, b.draws, b.indirect, b.triangles);
        }
    }));
    unhook.push(wrap(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', () => frame.dispatches++));
    unhook.push(wrap(GPUComputePassEncoder.prototype, 'dispatchWorkgroupsIndirect', () => frame.dispatches++));
    unhook.push(wrap(GPUCommandEncoder.prototype, 'beginRenderPass', (pass) => {
        frame.renderPasses++;
        bucket(byPass, pass.label || '(unlabelled pass)').passes++;
    }));
    unhook.push(wrap(GPUCommandEncoder.prototype, 'beginComputePass', () => frame.computePasses++));
}

// ---- the overlay -------------------------------------------------------------------------------

const HISTORY = 120;
const history = { fps: [], cpu: [], gpu: [] };
let box = null;
let graph = null;
let fields = null;
let details = null;
let expanded = false;
let lastTick = 0;
let lastShown = 0;
// what the frames since the last refresh add up to
const fresh = () => ({ frames: 0, interval: 0, cpu: 0, draws: 0, indirect: 0, triangles: 0, dispatches: 0, renderPasses: 0, computePasses: 0, byPass: new Map(), byMaterial: new Map() });
let acc = fresh();
// the last refresh's totals, for report()
let last = null;
let profile = null;
// the cluster LOD cuts' triangles (camera, all views), from the renderer's cull readbacks
let clusters = { camera: 0, all: 0 };

const CSS = `
.kansei-perf { position: fixed; left: 8px; bottom: 28px; z-index: 30; width: 340px; padding: 6px 8px 4px; box-sizing: border-box;
  font: 11px/1.3 ui-monospace, Menlo, monospace; color: #dfe6f2; background: rgba(16, 18, 24, 0.86); border-radius: 6px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35); cursor: pointer; user-select: none; }
.kansei-perf .row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 2px 6px; }
.kansei-perf .cell { min-width: 0; }
.kansei-perf .k { color: #8a93a6; font-size: 9px; letter-spacing: 0.04em; text-transform: uppercase; }
.kansei-perf .v { font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kansei-perf .fps { color: #7ee08a; } .kansei-perf .cpu { color: #6cb6ff; } .kansei-perf .gpu { color: #ffb45e; }
.kansei-perf canvas { display: block; width: 100%; height: 44px; margin: 5px 0 3px; }
.kansei-perf .more { color: #8a93a6; font-size: 9px; text-align: right; }
.kansei-perf table { width: 100%; border-collapse: collapse; margin-top: 4px; font-size: 10.5px; table-layout: fixed; }
.kansei-perf td { padding: 0 0 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kansei-perf td.ms { text-align: right; width: 52px; }
.kansei-perf td.bar { width: 44px; padding-left: 6px; }
.kansei-perf td.bar i { display: block; height: 6px; background: #ffb45e; border-radius: 1px; }
.kansei-perf th.ms { text-align: right; }
.kansei-perf th { text-align: left; color: #8a93a6; font-weight: normal; font-size: 9px; text-transform: uppercase; padding-top: 5px; }
.kansei-perf .details { max-height: 46vh; overflow-y: auto; }
.kansei-perf.open { width: 460px; }
`;

function mount() {
    if (box || !hasDom || !document.body) return;
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    box = document.createElement('div');
    box.className = 'kansei-perf';
    box.title = 'click: passes and counts · ` hides';
    const cell = (key, label, cls = '') => `<div class="cell"><div class="k">${label}</div><div class="v ${cls}" data-k="${key}">-</div></div>`;
    box.innerHTML = `<div class="row">${cell('fps', 'fps', 'fps')}${cell('cpu', 'cpu ms', 'cpu')}${cell('gpu', 'gpu ms', 'gpu')}${cell('mem', 'gpu mem')}</div>`
        + `<div class="row">${cell('draws', 'draws')}${cell('tris', 'triangles')}${cell('dispatches', 'dispatches')}${cell('pipelines', 'pipelines')}</div>`
        + `<canvas></canvas><div class="more">▸ passes</div><div class="details" hidden></div>`;
    document.body.appendChild(box);
    graph = box.querySelector('canvas');
    details = box.querySelector('.details');
    fields = Object.fromEntries([...box.querySelectorAll('[data-k]')].map((el) => [el.dataset.k, el]));
    box.addEventListener('click', () => {
        expanded = !expanded;
        details.hidden = !expanded;
        box.classList.toggle('open', expanded);
        box.querySelector('.more').textContent = expanded ? '▾ passes' : '▸ passes';
        refresh(true);
    });
}

const mb = (bytes) => (bytes >= 1 << 30 ? `${(bytes / (1 << 30)).toFixed(2)} GB` : `${(bytes / (1 << 20)).toFixed(0)} MB`);
const count = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e5 ? `${(n / 1e3).toFixed(0)}k` : n >= 1e4 ? `${(n / 1e3).toFixed(1)}k` : `${Math.round(n)}`);
const escape = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);

function drawGraph() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(graph.clientWidth * dpr);
    const h = Math.round(graph.clientHeight * dpr);
    if (graph.width !== w || graph.height !== h) { graph.width = w; graph.height = h; }
    const g = graph.getContext('2d');
    g.clearRect(0, 0, w, h);
    // ms on one scale (33 ms at least, the frame budgets marked), fps on its own
    const msMax = Math.max(33.4, ...history.cpu, ...history.gpu) * 1.1;
    const fpsMax = Math.max(60, ...history.fps) * 1.1;
    g.strokeStyle = 'rgba(255,255,255,0.12)';
    g.lineWidth = 1;
    for (const budget of [16.7, 33.3]) {
        const y = h - (budget / msMax) * h;
        g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
    }
    const line = (values, max, color) => {
        if (values.length < 2) return;
        g.strokeStyle = color;
        g.lineWidth = 1.5 * dpr;
        g.beginPath();
        values.forEach((v, i) => {
            const x = (i / (HISTORY - 1)) * w;
            const y = h - Math.min(v / max, 1) * (h - 2) - 1;
            i ? g.lineTo(x, y) : g.moveTo(x, y);
        });
        g.stroke();
    };
    line(history.fps, fpsMax, '#7ee08a');
    line(history.cpu, msMax, '#6cb6ff');
    line(history.gpu, msMax, '#ffb45e');
}

function table(title, rows, max) {
    const body = rows.map(([label, ms, note]) => `<tr><td title="${escape(label)}">${escape(label)}${note ? ` <span class="k">${note}</span>` : ''}</td><td class="ms">${ms.toFixed(2)}</td><td class="bar"><i style="width:${Math.min(100, (ms / max) * 100).toFixed(0)}%"></i></td></tr>`).join('');
    return `<tr><th colspan="3">${title}</th></tr>${body}`;
}

// a frame's draws and triangles per pass or material, most triangles first (passes: how many of
// the pass a frame)
function drawTable(title, map, n, passes) {
    const rows = [...map].sort((a, b) => b[1].triangles - a[1].triangles || b[1].draws - a[1].draws).slice(0, 24);
    const body = rows.map(([label, b]) => {
        const times = passes && b.passes / n > 1.05 ? ` <span class="k">x${(b.passes / n).toFixed(0)}</span>` : '';
        const indirect = b.indirect ? ` <span class="k">${count(b.indirect / n)} ind.</span>` : '';
        return `<tr><td title="${escape(label)}">${escape(label)}${times}${indirect}</td><td class="ms">${count(b.draws / n)}</td><td class="ms">${count(b.triangles / n)}</td></tr>`;
    }).join('');
    return `<tr><th>${title}</th><th class="ms">draws</th><th class="ms">tris</th></tr>${body}`;
}

// the largest entries of a label -> { count, bytes } map
function sizeTable(title, map, value, key) {
    const rows = [...map].sort((a, b) => key(b[1]) - key(a[1])).slice(0, 16);
    const body = rows.map(([label, e]) => `<tr><td title="${escape(label)}">${escape(label)}${e.count > 1 ? ` <span class="k">x${e.count}</span>` : ''}</td><td class="ms" colspan="2">${value(e)}</td></tr>`).join('');
    return `<tr><th colspan="3">${title}</th></tr>${body}`;
}

function refresh(force = false) {
    const now = performance.now();
    if (!force && now - lastShown < 250) return;
    const since = now - lastShown;
    lastShown = now;
    const n = Math.max(acc.frames, 1);
    const fps = acc.interval > 0 ? (1000 * acc.frames) / acc.interval : 0;
    const cpu = acc.cpu / n;
    const gpu = profile ? profile.gpu_ms : NaN;
    if (acc.frames) {
        for (const [key, value] of [['fps', fps], ['cpu', cpu], ['gpu', gpu]]) {
            if (Number.isFinite(value)) history[key].push(value);
            if (history[key].length > HISTORY) history[key].shift();
        }
    }
    fields.fps.textContent = fps ? fps.toFixed(0) : '-';
    fields.cpu.textContent = cpu ? cpu.toFixed(2) : '-';
    fields.gpu.textContent = Number.isFinite(gpu) ? gpu.toFixed(2) : '-';
    fields.gpu.title = profile ? `passes ${profile.gpu_ms.toFixed(2)} ms, first to last ${profile.gpu_span_ms.toFixed(2)} ms` : 'the page forwards no profile (kansei_wasm::perf::profile)';
    fields.mem.textContent = mb(memory.bufferBytes + memory.textureBytes);
    fields.draws.textContent = count(acc.draws / n);
    fields.draws.title = `${count(acc.indirect / n)} of them indirect`;
    const direct = acc.triangles / n;
    // (+: there are indirect draws besides, a compute pass's mesh such as a fluid's surface,
    // whose triangles only the GPU knows)
    fields.tris.textContent = count(direct + clusters.all) + (acc.indirect ? '+' : '');
    fields.tris.title = `direct draws ${count(direct)}, cluster LOD cuts ${count(clusters.all)} (camera ${count(clusters.camera)}); +: other indirect draws`;
    fields.dispatches.textContent = count(acc.dispatches / n);
    fields.pipelines.textContent = count(memory.renderPipelines + memory.computePipelines);
    drawGraph();
    if (expanded) {
        const passes = profile ? profile.passes.slice(0, 40) : [];
        const max = Math.max(1, ...passes.map((p) => p[1]));
        const cpuRows = profile ? profile.cpu.slice(0, 12) : [];
        const cpuMax = Math.max(1, ...cpuRows.map((c) => c[1]));
        const counts = [
            ['render passes', acc.renderPasses / n, `${count(acc.computePasses / n)} compute`],
            ['indirect draws', acc.indirect / n],
            ['buffers', memory.buffers, mb(memory.bufferBytes)],
            ['textures', memory.textures, mb(memory.textureBytes)],
            ['freed by the collector', memory.collected / (1 << 20), 'MB, dropped undestroyed'],
            ['render pipelines', memory.renderPipelines, `${memory.computePipelines} compute`],
            ['pipelines made since', pipelinesMade, `${(since / 1000).toFixed(1)} s`],
            ['shader modules', memory.shaders],
            ['bind groups made', memory.bindGroups],
        ];
        const countRows = counts.map(([label, value, note]) => `<tr><td>${label}${note ? ` <span class="k">${note}</span>` : ''}</td><td class="ms">${count(value)}</td><td></td></tr>`).join('');
        details.innerHTML = `<table>${profile ? table(`GPU passes (${profile.gpu_frames} frames)`, passes.map(([l, ms, c]) => [l, ms, c > 1.05 ? `x${c.toFixed(0)}` : '']), max) : '<tr><th colspan="3">GPU passes: none forwarded</th></tr>'}`
            + `${cpuRows.length ? table('CPU sections', cpuRows.map(([l, ms]) => [l, ms]), cpuMax) : ''}`
            + `<tr><th colspan="3">cluster LOD cuts (indirect)</th></tr><tr><td>camera / all views</td><td class="ms">${count(clusters.camera)}</td><td class="ms">${count(clusters.all)}</td></tr>`
            + drawTable('draws by pass', acc.byPass, n, true)
            + drawTable('draws by material', acc.byMaterial, n, false)
            + `<tr><th colspan="3">counts (per frame, or alive)</th></tr>${countRows}`
            + sizeTable('memory by label', memoryByLabel, (e) => mb(e.bytes), (e) => e.bytes)
            + sizeTable('pipelines by material', pipelinesByMaterial, (e) => `${e.count}`, (e) => e.count)
            + '</table>';
    }
    last = { ...acc, fps, cpu, gpu, pipelinesMade, since };
    acc = fresh();
    pipelinesMade = 0;
}

function show(on) {
    visible = !!on && hasDom;
    if (visible) mount();
    if (box) box.hidden = !visible;
    hookDraws(visible);
    lastTick = 0;
}

// ---- the Rust side (kansei_wasm::perf) ---------------------------------------------------------

export function perfVisible() {
    return visible;
}

export function perfSetVisible(on) {
    if (!!on !== visible) show(on);
}

export function perfFrame(cpuMs) {
    if (!visible || !box) return;
    const now = performance.now();
    if (lastTick) {
        acc.frames++;
        acc.interval += now - lastTick;
        acc.cpu += cpuMs;
        for (const key of ['draws', 'indirect', 'triangles', 'dispatches', 'renderPasses', 'computePasses']) acc[key] += frame[key];
        for (const [from, to] of [[byPass, acc.byPass], [byMaterial, acc.byMaterial]]) {
            for (const [key, b] of from) {
                const a = bucket(to, key);
                for (const k of ['passes', 'draws', 'indirect', 'triangles']) a[k] += b[k];
            }
        }
    }
    for (const key of Object.keys(frame)) frame[key] = 0;
    byPass.clear();
    byMaterial.clear();
    lastTick = now;
    refresh();
}

export function perfClusters(camera, all) {
    clusters.camera = camera;
    clusters.all = all;
}

export function perfProfile(json) {
    try {
        profile = JSON.parse(json);
    } catch (_) {
        profile = null;
    }
}

if (hasDom) {
    // report(): what the overlay last showed, as data (per frame), for scripts and tests
    const report = () => {
        if (!last) return null;
        const n = Math.max(last.frames, 1);
        const per = (map) => Object.fromEntries([...map].sort((a, b) => b[1].triangles - a[1].triangles).map(([k, b]) => [k, { passes: b.passes / n, draws: b.draws / n, indirect: b.indirect / n, triangles: b.triangles / n }]));
        return {
            fps: last.fps, cpu_ms: last.cpu, gpu_ms: last.gpu, draws: last.draws / n, indirect: last.indirect / n, triangles: last.triangles / n, dispatches: last.dispatches / n,
            cluster_triangles: { ...clusters },
            render_passes: last.renderPasses / n, compute_passes: last.computePasses / n, pipelines_made: last.pipelinesMade, since_ms: last.since,
            memory: { ...memory }, by_pass: per(last.byPass), by_material: per(last.byMaterial),
            memory_by_label: Object.fromEntries([...memoryByLabel].sort((a, b) => b[1].bytes - a[1].bytes)),
            pipelines_by_material: Object.fromEntries([...pipelinesByMaterial].sort((a, b) => b[1].count - a[1].count).map(([k, e]) => [k, e.count])),
            passes_ms: profile ? profile.passes : [],
        };
    };
    window.kanseiPerf = { show: () => show(true), hide: () => show(false), toggle: () => show(!visible), get visible() { return visible; }, report };
    window.addEventListener('keydown', (e) => {
        if (e.key !== '`' || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
        const el = e.target;
        if (el instanceof HTMLElement && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
        show(!visible);
    });
    if (visible) {
        if (document.body) show(true);
        else addEventListener('DOMContentLoaded', () => show(true));
    }
}
