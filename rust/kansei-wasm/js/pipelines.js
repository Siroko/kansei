// kansei_wasm's pipeline warm-up (src/pipelines.rs). wgpu's web backend makes every pipeline with
// createRenderPipeline / createComputePipeline, which Chrome compiles one after another on its GPU
// process the first time a page shows them (cached per origin after that): a first visit draws
// almost nothing for seconds. This records what the page makes and, on a later load, makes it all
// again with the *Async calls before the engine gets its device, which the browser compiles in
// parallel; the engine's own calls for the same descriptors then find them made (Dawn keeps
// pipelines, shader modules and layouts by content).
//
// - Recording: `?record_pipelines=1` or recordPipelines(); then pipelinesManifest() (or
//   window.kanseiPipelines.manifest() / .download()) is the JSON of every shader module, bind
//   group layout, pipeline layout and pipeline made since, in order, objects named by index.
// - Warming: warmPipelines(json) before the renderer asks for its device: the next requestDevice
//   resolves once the manifest's pipelines are made on the new device (each one that fails is
//   skipped and counted); a device already made gets them at once.
// - In a worker the same hooks work on the worker's GPU objects.

const scope = typeof self !== 'undefined' ? self : globalThis;
const hasGpu = typeof GPUDevice !== 'undefined';
const query = typeof location !== 'undefined' ? new URLSearchParams(location.search) : null;

let recording = !!query && query.get('record_pipelines') === '1';
const ids = new WeakMap();
const records = { modules: [], groupLayouts: [], pipelineLayouts: [], pipelines: [] };
let pending = null;
const devices = [];
const stats = { replayed: 0, failed: 0, ms: 0 };

function wrap(proto, name, after) {
    const original = proto[name];
    if (!original) return;
    proto[name] = function (...args) {
        const out = original.apply(this, args);
        after.call(this, out, args);
        return out;
    };
}

function id(object) {
    return object === undefined || object === 'auto' ? object : ids.get(object);
}

// A descriptor's stage, its module by index.
function stage(s) {
    if (!s) return s;
    const out = { ...s, module: id(s.module) };
    if (out.constants) out.constants = { ...out.constants };
    return out;
}

function note(list, object, desc) {
    ids.set(object, list.length);
    list.push(desc);
}

if (hasGpu) {
    wrap(GPUDevice.prototype, 'createShaderModule', function (module, [desc]) {
        if (recording) note(records.modules, module, { code: desc.code });
    });
    wrap(GPUDevice.prototype, 'createBindGroupLayout', function (layout, [desc]) {
        if (recording) note(records.groupLayouts, layout, { entries: JSON.parse(JSON.stringify(desc.entries)) });
    });
    wrap(GPUDevice.prototype, 'createPipelineLayout', function (layout, [desc]) {
        if (recording) note(records.pipelineLayouts, layout, { bindGroupLayouts: [...desc.bindGroupLayouts].map((l) => (l ? id(l) : null)) });
    });
    const pipeline = (kind) =>
        function (_, [desc]) {
            if (!recording) return;
            const d = { kind, layout: id(desc.layout) };
            if (kind === 'compute') d.compute = stage(desc.compute);
            else {
                d.vertex = stage(desc.vertex);
                d.vertex.buffers = JSON.parse(JSON.stringify(desc.vertex.buffers || []));
                if (desc.fragment) {
                    d.fragment = stage(desc.fragment);
                    d.fragment.targets = JSON.parse(JSON.stringify(desc.fragment.targets || []));
                }
                for (const k of ['primitive', 'depthStencil', 'multisample']) if (desc[k]) d[k] = JSON.parse(JSON.stringify(desc[k]));
            }
            // (only those whose objects were all recorded)
            const known = (v) => v !== undefined;
            const ok = (d.layout === 'auto' || known(d.layout)) && known((d.compute || d.vertex).module) && (!d.fragment || known(d.fragment.module));
            if (ok) records.pipelines.push(d);
        };
    wrap(GPUDevice.prototype, 'createComputePipeline', pipeline('compute'));
    wrap(GPUDevice.prototype, 'createRenderPipeline', pipeline('render'));

    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...args) {
        const device = await requestDevice.apply(this, args);
        devices.push(device);
        if (pending) await replay(device, pending);
        return device;
    };
}

// Make the manifest's objects on `device`, its pipelines with the *Async calls, all at once.
async function replay(device, manifest) {
    const t0 = performance.now();
    const was = recording;
    // (what the replay makes isn't recorded again)
    recording = false;
    const modules = manifest.modules.map((m) => device.createShaderModule(m));
    const groupLayouts = manifest.groupLayouts.map((g) => device.createBindGroupLayout(g));
    const pipelineLayouts = manifest.pipelineLayouts.map((p) => device.createPipelineLayout({ bindGroupLayouts: p.bindGroupLayouts.map((i) => (i === null ? null : groupLayouts[i])) }));
    recording = was;
    const layout = (i) => (i === 'auto' ? 'auto' : pipelineLayouts[i]);
    const withModule = (s) => (s ? { ...s, module: modules[s.module] } : s);
    const made = manifest.pipelines.map((p) => {
        const desc = { layout: layout(p.layout) };
        if (p.kind === 'compute') {
            desc.compute = withModule(p.compute);
            return device.createComputePipelineAsync(desc);
        }
        desc.vertex = withModule(p.vertex);
        if (p.fragment) desc.fragment = withModule(p.fragment);
        for (const k of ['primitive', 'depthStencil', 'multisample']) if (p[k]) desc[k] = p[k];
        return device.createRenderPipelineAsync(desc);
    });
    const results = await Promise.allSettled(made);
    stats.replayed += results.filter((r) => r.status === 'fulfilled').length;
    stats.failed += results.filter((r) => r.status === 'rejected').length;
    stats.ms += performance.now() - t0;
    // (the pipelines stay referenced by the device's cache only as long as these do)
    scope.__kanseiWarmed = (scope.__kanseiWarmed || []).concat(results.map((r) => r.value).filter(Boolean));
}

export function recordPipelines() {
    recording = true;
}

export function pipelinesManifest() {
    return JSON.stringify(records);
}

export async function warmPipelines(json) {
    const manifest = JSON.parse(json);
    pending = manifest;
    for (const device of devices) await replay(device, manifest);
}

export function warmStats() {
    return JSON.stringify(stats);
}

scope.kanseiPipelines = {
    manifest: pipelinesManifest,
    record: recordPipelines,
    stats: () => ({ ...stats }),
    download(name = 'pipelines.json') {
        if (typeof document === 'undefined') return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([pipelinesManifest()], { type: 'application/json' }));
        a.download = name;
        a.click();
    },
};
