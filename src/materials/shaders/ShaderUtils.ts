import { ShaderChunks } from "./ShaderChunks";

/**
 * Parses a shader string and replaces all #include directives with the corresponding shader chunks.
 * 
 * @param {string} string - The shader string containing #include directives.
 * @returns {string} - The shader string with all #include directives replaced.
 */
export const parseIncludes = (string: string): string => {
    const includePattern = /#include\s*<([^>]+)>/g;
    return string.replace(includePattern, (_, includePath) => replaceInclude(includePath));
}

const chunkMap = new Map();

/**
 * Replaces an #include directive with the corresponding shader chunk.
 * 
 * @param {string} includePath - The path of the shader chunk to include.
 * @returns {string} - The shader chunk corresponding to the include path.
 * @throws Will throw an error if the include path cannot be resolved.
 */
export const replaceInclude = (includePath: string): string => {
    let chunk = ShaderChunks[includePath as keyof typeof ShaderChunks];

    if (chunk === undefined) {
        const newInclude = chunkMap.get(includePath);
        if (newInclude !== undefined) {
            chunk = ShaderChunks[newInclude as keyof typeof ShaderChunks];
        } else {
            throw new Error('Can not resolve #include <' + includePath + '>');
        }
    }

    return parseIncludes(chunk);
}
