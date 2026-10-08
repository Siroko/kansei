/** A WGSL file imported as its source text (Vite `?raw`), as `materials/shaders/SharedWGSL.ts` imports the Rust engine's shaders. */
declare module '*.wgsl?raw' {
    const source: string;
    export default source;
}

/** An asset imported as its URL (Vite `?url`): a data URL in the library build, as the KTX2 transcoder's wasm. */
declare module '*.wasm?url' {
    const url: string;
    export default url;
}
