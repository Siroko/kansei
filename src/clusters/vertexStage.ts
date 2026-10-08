import clusterMeshWgsl from '../../rust/kansei-core/src/shaders/cluster_mesh.wgsl?raw';

/**
 * The vertex stage of a material's cluster pipeline, generated from its WGSL (Rust
 * `clusters/vertex_stage.rs`). `vertex_main` becomes a plain function.
 * `kansei_cluster_vertex_main` finds its draw-list entry (from the cut's index:
 * `entry << 8 | the vertex's index in the cluster`), reads the mesh's vertex and the instance
 * record's attributes (from the instance buffer's layout), fills `vertex_main`'s inputs, and
 * calls it.
 */

/**
 * A cluster mesh's words (`ClusterMesh.gpuWords`) in `kansei_cluster_mesh`, which the including
 * module binds: readers of its clusters, levels and vertices. Rust: `CLUSTER_MESH_WGSL`.
 */
export const CLUSTER_MESH_WGSL: string = clusterMeshWgsl;

/** The generated entry point. */
export const CLUSTER_VERTEX_ENTRY = 'kansei_cluster_vertex_main';

/**
 * The generated stage's group 2 bindings, after the mesh matrices at 0 and 1: the packed mesh,
 * the view's draw list and the instance records; and the readers of 1-4 components, missing
 * ones (0, 0, 0, 1) as a vertex fetch fills them.
 */
const PRELUDE = `
@group(2) @binding(2) var<storage, read> kansei_cluster_mesh: array<u32>;
@group(2) @binding(3) var<storage, read> kansei_cluster_draws: array<vec2<u32>>;
@group(2) @binding(4) var<storage, read> kansei_cluster_records: array<u32>;

fn kansei_cluster_attribute(vertex: u32, word: u32, count: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    for (var i = 0u; i < count; i++) {
        v[i] = kansei_vertex_f32(vertex, word + i);
    }
    return v;
}

fn kansei_record_f32(record: u32, word: u32, count: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    for (var i = 0u; i < count; i++) {
        v[i] = bitcast<f32>(kansei_cluster_records[record * KANSEI_RECORD_WORDS + word + i]);
    }
    return v;
}

fn kansei_record_u32(record: u32, word: u32, count: u32) -> vec4<u32> {
    var v = vec4<u32>(0u, 0u, 0u, 1u);
    for (var i = 0u; i < count; i++) {
        v[i] = kansei_cluster_records[record * KANSEI_RECORD_WORDS + word + i];
    }
    return v;
}

fn kansei_record_i32(record: u32, word: u32, count: u32) -> vec4<i32> {
    var v = vec4<i32>(0, 0, 0, 1);
    for (var i = 0u; i < count; i++) {
        v[i] = bitcast<i32>(kansei_cluster_records[record * KANSEI_RECORD_WORDS + word + i]);
    }
    return v;
}
`;

/** An input of `vertex_main`: its location, name and WGSL type. */
interface Input {
    location: number;
    name: string;
    ty: string;
}

/** The instance records' layout: the geometry's one instance buffer's vertex layout. */
export interface InstanceLayout {
    arrayStride: number;
    attributes: Iterable<GPUVertexAttribute>;
}

/**
 * `code` (a material's WGSL, includes resolved) with the cluster vertex stage, for instance
 * records laid out as `instances` (the geometry's one instance buffer, if any). Throws, saying
 * what the stage can't feed: a builtin input, a location no buffer provides, or a type its
 * buffer's format doesn't match.
 */
export function clusterVertexStage(source: string, instances: InstanceLayout | null): string {
    return generate(source, instances, false);
}

/** The debug view's vertex body: the generated stage as a plain function of the cut's index. */
export const CLUSTER_VERTEX_FN = 'kansei_cluster_vertex';

/**
 * `clusterVertexStage`'s code with the stage as a plain function instead of an entry point,
 * `kansei_cluster_vertex(cut index, instance index) -> vertex_main's output`, for the cluster
 * debug view (`ClusterDebug`), which reads the cut's index itself; and the name of the
 * `@builtin(position)` member of that output (null when the output is the position itself).
 * Throws as `clusterVertexStage` does.
 */
export function clusterVertexFunction(source: string, instances: InstanceLayout | null): { code: string; position: string | null } {
    const code = generate(source, instances, true);
    const stripped = stripComments(source);
    return { code, position: positionMember(stripped, stripAttributes(vertexReturns(stripped))) };
}

/** What `vertex_main` returns, attributes and all. */
function vertexReturns(code: string): string {
    const name = findFn(code, 'vertex_main')!;
    const close = matchingParen(code, code.indexOf('(', name))!;
    return code.slice(close + 1, code.indexOf('{', close)).trim().slice(2).trim();
}

/** The member of `struct <ty>` marked `@builtin(position)`, or null when `ty` is no struct. */
function positionMember(code: string, ty: string): string | null {
    for (let at = code.indexOf('struct'); at >= 0; at = code.indexOf('struct', at + 1)) {
        const rest = code.slice(at + 'struct'.length).trimStart();
        if (!rest.startsWith(ty) || isIdent(rest[ty.length])) continue;
        const after = rest.slice(ty.length).trimStart();
        if (!after.startsWith('{')) continue;
        const start = code.length - after.length;
        const end = code.indexOf('}', start);
        for (const member of splitTopLevel(code.slice(start + 1, end), ',')) {
            if (!/@builtin\s*\(\s*position\s*\)/.test(member)) continue;
            const name = stripAttributes(member);
            return name.slice(0, name.indexOf(':')).trim();
        }
        throw new Error(`${ty} has no @builtin(position) member`);
    }
    return null;
}

function generate(source: string, instances: InstanceLayout | null, plain: boolean): string {
    const code = stripComments(source);
    const name = findFn(code, 'vertex_main');
    if (name === null) throw new Error('no fn vertex_main');
    const attributed = code.slice(0, name).trimEnd();
    if (!attributed.endsWith('@vertex')) throw new Error('vertex_main is not marked @vertex');
    const vertexAttribute = attributed.length - '@vertex'.length;
    const openAt = code.indexOf('(', name);
    if (openAt < 0) throw new Error('vertex_main has no parameter list');
    const close = matchingParen(code, openAt);
    if (close === null) throw new Error("vertex_main's parameters are unbalanced");
    const body = code.indexOf('{', close);
    if (body < 0) throw new Error('vertex_main has no body');
    const arrow = code.slice(close + 1, body).trim();
    if (!arrow.startsWith('->')) throw new Error('vertex_main returns nothing');
    const returns = arrow.slice(2).trim();
    const params = splitTopLevel(code.slice(openAt + 1, close), ',');

    let plainParams: string, fill = '', args: string;
    if (params.length === 1 && !params[0].startsWith('@')) {
        // one struct of located members
        const colon = params[0].indexOf(':');
        if (colon < 0) throw new Error("vertex_main's parameter has no type");
        const arg = params[0].slice(0, colon).trim();
        const ty = params[0].slice(colon + 1).trim();
        const members = structMembers(code, ty);
        if (members === null) throw new Error(`vertex_main's input struct ${ty} not found`);
        fill = `    var kansei_input: ${ty};\n`;
        for (const member of members) {
            fill += `    kansei_input.${member.name} = ${inputExpression(member, instances)};\n`;
        }
        plainParams = `${arg}: ${ty}`;
        args = 'kansei_input';
    } else {
        const inputs = params.map(parseInput);
        plainParams = inputs.map((i) => `${i.name}: ${i.ty}`).join(', ');
        args = inputs.map((i) => inputExpression(i, instances)).join(', ');
    }

    const recordWords = instances ? instances.arrayStride / 4 : 0;
    let out = code.slice(0, vertexAttribute);
    out += code.slice(name, openAt + 1);
    out += plainParams;
    out += `) -> ${stripAttributes(returns)} `;
    out += code.slice(body);
    out += `\nconst KANSEI_RECORD_WORDS: u32 = ${recordWords}u;\n${PRELUDE}\n${CLUSTER_MESH_WGSL}\n`;
    out += plain
        ? `fn ${CLUSTER_VERTEX_FN}(kansei_vertex_index: u32, kansei_instance_index: u32) -> ${stripAttributes(returns)} {\n`
        : `@vertex\nfn ${CLUSTER_VERTEX_ENTRY}(@builtin(vertex_index) kansei_vertex_index: u32, @builtin(instance_index) kansei_instance_index: u32) -> ${returns} {\n`;
    // (the cut's index: its draw-list entry, and the vertex's index in the entry's cluster)
    out += '    let kansei_draw = kansei_cluster_draws[kansei_vertex_index >> 8u];\n';
    out += '    let kansei_vertex = kansei_cluster_local_vertex(kansei_draw.y, kansei_vertex_index & 0xffu);\n';
    out += '    let kansei_record = kansei_draw.x;\n';
    out += fill;
    out += `    return vertex_main(${args});\n}\n`;
    return out;
}

/** `code` without its `//` and (nested) `/* *\/` comments. */
function stripComments(code: string): string {
    let out = '';
    let i = 0, depth = 0;
    while (i < code.length) {
        if (code.startsWith('/*', i)) {
            depth++;
            i += 2;
        } else if (depth > 0 && code.startsWith('*/', i)) {
            depth--;
            i += 2;
            out += ' ';
        } else if (depth > 0) {
            i++;
        } else if (code.startsWith('//', i)) {
            while (i < code.length && code[i] !== '\n') i++;
        } else {
            out += code[i++];
        }
    }
    return out;
}

const isIdent = (c: string | undefined) => c !== undefined && /[\p{L}\p{N}_]/u.test(c);

/** Where `fn <name>` starts: the index of its `fn`. */
function findFn(code: string, name: string): number | null {
    for (let at = code.indexOf(name); at >= 0; at = code.indexOf(name, at + 1)) {
        const before = code.slice(0, at).trimEnd();
        if (!before.endsWith('fn')) continue;
        const fnAt = before.length - 2;
        const boundaryBefore = !isIdent(code[fnAt - 1]);
        const boundaryAfter = code.length > at + name.length && !isIdent(code[at + name.length]);
        if (boundaryBefore && boundaryAfter && before.length < at) return fnAt;
    }
    return null;
}

/** The `)` closing the `(` at `open`. */
function matchingParen(code: string, open: number): number | null {
    let depth = 0;
    for (let i = open; i < code.length; i++) {
        if (code[i] === '(') depth++;
        else if (code[i] === ')') {
            depth--;
            if (depth === 0) return i;
        }
    }
    return null;
}

/** `text` split at `separator` outside brackets, trimmed, empty pieces (a trailing separator) dropped. */
function splitTopLevel(text: string, separator: string): string[] {
    const pieces: string[] = [];
    let current = '', depth = 0;
    for (const c of text) {
        if (c === '(' || c === '<' || c === '[') depth++;
        else if (c === ')' || c === '>' || c === ']') depth--;
        if (c === separator && depth === 0) {
            pieces.push(current);
            current = '';
        } else {
            current += c;
        }
    }
    pieces.push(current);
    return pieces.map((p) => p.trim()).filter((p) => p.length > 0);
}

/** The members of `struct <name>`, or null when there is no such struct. */
function structMembers(code: string, name: string): Input[] | null {
    for (let at = code.indexOf('struct'); at >= 0; at = code.indexOf('struct', at + 1)) {
        const rest = code.slice(at + 'struct'.length).trimStart();
        if (!rest.startsWith(name)) continue;
        const after = rest.slice(name.length).trimStart();
        if (!after.startsWith('{')) continue;
        const start = code.length - after.length;
        const end = code.indexOf('}', start);
        if (end < 0) return null;
        return splitTopLevel(code.slice(start + 1, end), ',').map(parseInput);
    }
    return null;
}

/** `@location(n) name: type` (other attributes skipped); a builtin throws. */
function parseInput(text: string): Input {
    let rest = text.trim();
    let location: number | null = null;
    while (rest.startsWith('@')) {
        const after = rest.slice(1);
        const nameEnd = after.search(/[^\p{L}\p{N}_]/u);
        const attribute = nameEnd < 0 ? after : after.slice(0, nameEnd);
        let tail = nameEnd < 0 ? '' : after.slice(nameEnd);
        let argument = '';
        if (tail.trimStart().startsWith('(')) {
            const open = tail.indexOf('(');
            const close = matchingParen(tail, open);
            if (close === null) throw new Error(`unbalanced attribute in \`${text}\``);
            argument = tail.slice(open + 1, close).trim();
            tail = tail.slice(close + 1);
        }
        if (attribute === 'builtin') {
            throw new Error(`vertex_main reads the builtin ${argument} (\`${text}\`), which the cluster path doesn't provide`);
        }
        if (attribute === 'location') {
            if (!/^\d+$/.test(argument)) throw new Error(`bad location in \`${text}\``);
            location = Number(argument);
        }
        rest = tail.trimStart();
    }
    const colon = rest.indexOf(':');
    if (colon < 0) throw new Error(`no type in \`${text}\``);
    if (location === null) throw new Error(`\`${text}\` has no @location`);
    return { location, name: rest.slice(0, colon).trim(), ty: rest.slice(colon + 1).split(/\s+/).join('') };
}

/** `text` without its `@attribute(...)`s. */
function stripAttributes(text: string): string {
    let rest = text.trim();
    while (rest.startsWith('@')) {
        const after = rest.slice(1);
        const nameEnd = after.search(/[^\p{L}\p{N}_]/u);
        let tail = nameEnd < 0 ? '' : after.slice(nameEnd);
        if (tail.trimStart().startsWith('(')) {
            const open = tail.indexOf('(');
            const close = matchingParen(tail, open);
            tail = tail.slice(close === null ? tail.length : close + 1);
        }
        rest = tail.trimStart();
    }
    return rest;
}

type Kind = 'f' | 'u' | 'i';

/** A WGSL input type's components and scalar kind. */
function shape(ty: string): [number, Kind] | null {
    const scalar = (s: string): Kind | null => ({ f32: 'f', f: 'f', u32: 'u', u: 'u', i32: 'i', i: 'i' } as Record<string, Kind>)[s] ?? null;
    const kind = scalar(ty);
    if (kind) return [1, kind];
    if (!ty.startsWith('vec')) return null;
    const n = Number(ty[3]);
    if (!(n >= 2 && n <= 4)) return null;
    const element = scalar(ty.slice(4).replace(/^<+/, '').replace(/>+$/, ''));
    return element ? [n, element] : null;
}

/** A 32-bit vertex format's components and scalar kind. */
function formatShape(format: GPUVertexFormat): [number, Kind] | null {
    const m = /^(float|uint|sint)32(?:x([234]))?$/.exec(format);
    if (!m) return null;
    return [m[2] ? Number(m[2]) : 1, ({ float: 'f', uint: 'u', sint: 'i' } as const)[m[1] as 'float' | 'uint' | 'sint']];
}

/**
 * The WGSL expression that feeds `input`: the mesh vertex's (locations 0-2, the standard vertex)
 * or the instance record's attribute at its location.
 */
function inputExpression(input: Input, instances: InstanceLayout | null): string {
    const location = input.location;
    const s = shape(input.ty);
    if (!s) throw new Error(`@location(${location}) ${input.name}: ${input.ty} isn't a 32-bit scalar or vector`);
    const [count, kind] = s;
    let expression: string, provided: Kind;
    if (location === 0) [expression, provided] = ['kansei_cluster_attribute(kansei_vertex, 0u, 4u)', 'f'];
    else if (location === 1) [expression, provided] = ['kansei_cluster_attribute(kansei_vertex, 4u, 3u)', 'f'];
    else if (location === 2) [expression, provided] = ['kansei_cluster_attribute(kansei_vertex, 7u, 2u)', 'f'];
    else {
        const attribute = instances ? [...instances.attributes].find((a) => a.shaderLocation === location) : undefined;
        if (!attribute) throw new Error(`vertex_main reads @location(${location}), which no buffer provides`);
        const fs = formatShape(attribute.format);
        if (!fs) throw new Error(`@location(${location}): ${attribute.format} isn't a 32-bit format`);
        if (attribute.offset % 4 !== 0) throw new Error(`@location(${location}): offset ${attribute.offset} isn't a multiple of 4`);
        const reader = { f: 'kansei_record_f32', u: 'kansei_record_u32', i: 'kansei_record_i32' }[fs[1]];
        expression = `${reader}(kansei_record, ${attribute.offset / 4}u, ${fs[0]}u)`;
        provided = fs[1];
    }
    if (kind !== provided) {
        throw new Error(`@location(${location}) is ${input.ty} in vertex_main, but its buffer holds ${{ f: 'floats', u: 'u32s', i: 'i32s' }[provided]}`);
    }
    return count === 1 ? `${expression}.x` : count === 2 ? `${expression}.xy` : count === 3 ? `${expression}.xyz` : expression;
}
