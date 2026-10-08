/** A WGSL file imported as its source text (Vite `?raw`), as `materials/shaders/SharedWGSL.ts` imports the Rust engine's shaders. */
declare module '*.wgsl?raw' {
    const source: string;
    export default source;
}
