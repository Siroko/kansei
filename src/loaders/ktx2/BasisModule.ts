/**
 * Instantiates the vendored Basis Universal transcoder (`basis/`). `Basis.ts` imports this module
 * dynamically: the library build inlines the WebAssembly here (Vite library mode inlines assets),
 * so it is downloaded only by pages that load a KTX2 file.
 */
import BASIS from "./basis/basis_transcoder.js";

/** The initialized Emscripten module (`KTX2File`, `isFormatSupported`, ...). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function createBasisModule(): Promise<any> {
    const wasmUrl = new URL("./basis/basis_transcoder.wasm", import.meta.url).href;
    const module = await BASIS({ locateFile: () => wasmUrl });
    module.initializeBasis();
    return module;
}
