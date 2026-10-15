// The canvas's colour on HDR screens (`Canvas::renderer`): there the renderer draws to an
// rgba16float canvas, which the compositor takes without an 8-bit quantize.

// Whether the screen shows high dynamic range (in a worker, the page's screen as posted).
export function screenIsHdr(element) {
    if (typeof element?.dynamicRangeHigh === 'boolean') return element.dynamicRangeHigh;
    return globalThis.matchMedia?.('(dynamic-range: high)').matches ?? false;
}

// wgpu configures the context without a tone mapping, so a float canvas would be clamped to SDR
// ("standard"): ask for "extended" for every rgba16float configure (a resize configures again).
export function extendFloatCanvases() {
    const proto = globalThis.GPUCanvasContext?.prototype;
    if (!proto || proto.kanseiExtended) return;
    const configure = proto.configure;
    proto.configure = function (config) {
        if (config.format === 'rgba16float' && !config.toneMapping) {
            config.toneMapping = { mode: 'extended' };
        }
        return configure.call(this, config);
    };
    proto.kanseiExtended = true;
}
