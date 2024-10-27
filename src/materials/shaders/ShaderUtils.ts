import { ShaderChunks } from "./ShaderChunks";

// Resolve Includes
export const parseIncludes = (string: string) => {
    const includePattern = /#include\s*<([^>]+)>/g;
    return string.replace(includePattern, (_, includePath) => replaceInclude(includePath));
}

const chunkMap = new Map();
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
