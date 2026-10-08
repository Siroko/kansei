/** The Emscripten factory of Binomial's Basis Universal transcoder (see README.md). */
declare const BASIS: (moduleArg?: { locateFile?: (path: string, prefix: string) => string; wasmBinary?: ArrayBuffer }) => Promise<any>;
export default BASIS;
