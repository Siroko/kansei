/**
 * Instantiates the vendored Basis Universal transcoder (`basis/`). `Basis.ts` imports this module
 * dynamically: the library build inlines the WebAssembly here as a data URL string (Vite library
 * mode inlines assets), so it is downloaded only by pages that load a KTX2 file. It is a `?url`
 * import, not `new URL(..., import.meta.url)`: Vite's dev server rewrites a data URL in that
 * pattern into a path under its deps folder when an app pre-bundles the package.
 */
import BASIS from "./basis/basis_transcoder.js";
import wasmUrl from "./basis/basis_transcoder.wasm?url";

/** The initialized Emscripten module (`KTX2File`, `isFormatSupported`, ...). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function createBasisModule(): Promise<any> {
    const module = await BASIS({ locateFile: () => wasmUrl });
    module.initializeBasis();
    return module;
}
