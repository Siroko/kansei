//! The vertex stage of a material's cluster pipeline, generated from its WGSL. `vertex_main`
//! becomes a plain function. `kansei_cluster_vertex_main` finds its draw-list entry (from the
//! cut's index: `entry << 8 | the vertex's index in the cluster`), reads the
//! mesh's vertex and the instance record's attributes (from the instance buffer's layout),
//! fills `vertex_main`'s inputs, and calls it.

use super::gpu::CLUSTER_MESH_WGSL;
use crate::buffers::InstanceBufferLayout;

/// The generated entry point.
pub(crate) const CLUSTER_VERTEX_ENTRY: &str = "kansei_cluster_vertex_main";

/// The generated stage's group 2 bindings, after the mesh matrices at 0 and 1: the packed mesh,
/// the view's draw list and the instance records; and the readers of 1-4 components, missing
/// ones (0, 0, 0, 1) as a vertex fetch fills them.
const PRELUDE: &str = r#"
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
"#;

/// An input of `vertex_main`: its location, name and WGSL type.
struct Input {
    location: u32,
    name: String,
    ty: String,
}

/// `code` (a material's WGSL, includes resolved) with the cluster vertex stage, for instance
/// records laid out as `instances` (the geometry's one instance buffer, if any). The Err says
/// what the stage can't feed: a builtin input, a location no buffer provides, or a type its
/// buffer's format doesn't match.
pub(crate) fn cluster_vertex_stage(code: &str, instances: Option<&InstanceBufferLayout>) -> Result<String, String> {
    let code = strip_comments(code);
    let name = find_fn(&code, "vertex_main").ok_or("no fn vertex_main")?;
    let attributed = code[..name].trim_end();
    let vertex_attribute = attributed.strip_suffix("@vertex").ok_or("vertex_main is not marked @vertex")?.len();
    let open = name + code[name..].find('(').ok_or("vertex_main has no parameter list")?;
    let close = matching_paren(&code, open).ok_or("vertex_main's parameters are unbalanced")?;
    let body = close + code[close..].find('{').ok_or("vertex_main has no body")?;
    let returns = code[close + 1..body].trim().strip_prefix("->").ok_or("vertex_main returns nothing")?.trim().to_string();
    let params = split_top_level(&code[open + 1..close], ',');

    let (plain_params, fill, args) = if params.len() == 1 && !params[0].starts_with('@') {
        // one struct of located members
        let (arg, ty) = params[0].split_once(':').ok_or("vertex_main's parameter has no type")?;
        let ty = ty.trim();
        let members = struct_members(&code, ty).ok_or_else(|| format!("vertex_main's input struct {ty} not found"))?;
        let mut fill = format!("    var kansei_input: {ty};\n");
        for member in members? {
            fill += &format!("    kansei_input.{} = {};\n", member.name, input_expression(&member, instances)?);
        }
        (format!("{}: {ty}", arg.trim()), fill, "kansei_input".to_string())
    } else {
        let inputs = params.iter().map(|p| parse_input(p)).collect::<Result<Vec<_>, _>>()?;
        let plain = inputs.iter().map(|i| format!("{}: {}", i.name, i.ty)).collect::<Vec<_>>().join(", ");
        let args = inputs.iter().map(|i| input_expression(i, instances)).collect::<Result<Vec<_>, _>>()?.join(", ");
        (plain, String::new(), args)
    };

    let record_words = instances.map_or(0, |l| l.stride / 4);
    let mut out = String::with_capacity(code.len() + PRELUDE.len() + CLUSTER_MESH_WGSL.len() + 1024);
    out += &code[..vertex_attribute];
    out += &code[name..open + 1];
    out += &plain_params;
    out += &format!(") -> {} ", strip_attributes(&returns));
    out += &code[body..];
    out += &format!("\nconst KANSEI_RECORD_WORDS: u32 = {record_words}u;\n{PRELUDE}\n{CLUSTER_MESH_WGSL}\n");
    out += &format!("@vertex\nfn {CLUSTER_VERTEX_ENTRY}(@builtin(vertex_index) kansei_vertex_index: u32, @builtin(instance_index) kansei_instance_index: u32) -> {returns} {{\n");
    // (the cut's index: its draw-list entry, and the vertex's index in the entry's cluster)
    out += "    let kansei_draw = kansei_cluster_draws[kansei_vertex_index >> 8u];\n";
    out += "    let kansei_vertex = kansei_cluster_local_vertex(kansei_draw.y, kansei_vertex_index & 0xffu);\n";
    out += "    let kansei_record = kansei_draw.x;\n";
    out += &fill;
    out += &format!("    return vertex_main({args});\n}}\n");
    Ok(out)
}

/// `code` without its `//` and (nested) `/* */` comments.
fn strip_comments(code: &str) -> String {
    let bytes = code.as_bytes();
    let mut out = String::with_capacity(code.len());
    let (mut i, mut depth) = (0, 0);
    while i < bytes.len() {
        if bytes[i..].starts_with(b"/*") {
            depth += 1;
            i += 2;
        } else if depth > 0 && bytes[i..].starts_with(b"*/") {
            depth -= 1;
            i += 2;
            out.push(' ');
        } else if depth > 0 {
            i += 1;
        } else if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            let c = code[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// Where `fn <name>` starts: the byte index of its `fn`.
fn find_fn(code: &str, name: &str) -> Option<usize> {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    code.match_indices(name).find_map(|(at, _)| {
        let before = code[..at].trim_end();
        let fn_at = before.strip_suffix("fn")?.len();
        let boundary_before = code[..fn_at].chars().next_back().is_none_or(|c| !is_ident(c));
        let boundary_after = code[at + name.len()..].chars().next().is_some_and(|c| !is_ident(c));
        (boundary_before && boundary_after && before.len() < at).then_some(fn_at)
    })
}

/// The `)` closing the `(` at `open`.
fn matching_paren(code: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in code[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// `text` split at `separator` outside brackets, trimmed, empty pieces (a trailing separator)
/// dropped.
fn split_top_level(text: &str, separator: char) -> Vec<String> {
    let (mut pieces, mut current, mut depth) = (Vec::new(), String::new(), 0i32);
    for c in text.chars() {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            _ => {}
        }
        if c == separator && depth == 0 {
            pieces.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    pieces.push(current);
    pieces.into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}

/// The members of `struct <name>`, or None when there is no such struct.
fn struct_members(code: &str, name: &str) -> Option<Result<Vec<Input>, String>> {
    let start = code.match_indices("struct").find_map(|(at, _)| {
        let rest = code[at + "struct".len()..].trim_start();
        let after = rest.strip_prefix(name)?;
        after.trim_start().starts_with('{').then(|| code.len() - after.trim_start().len())
    })?;
    let end = start + code[start..].find('}')?;
    Some(split_top_level(&code[start + 1..end], ',').iter().map(|m| parse_input(m)).collect())
}

/// `@location(n) name: type` (other attributes skipped); a builtin is an Err.
fn parse_input(text: &str) -> Result<Input, String> {
    let mut rest = text.trim();
    let mut location = None;
    while let Some(after) = rest.strip_prefix('@') {
        let name_end = after.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(after.len());
        let (attribute, mut tail) = after.split_at(name_end);
        let mut argument = "";
        if tail.trim_start().starts_with('(') {
            let open = tail.find('(').unwrap();
            let close = matching_paren(tail, open).ok_or_else(|| format!("unbalanced attribute in `{text}`"))?;
            argument = tail[open + 1..close].trim();
            tail = &tail[close + 1..];
        }
        match attribute {
            "builtin" => return Err(format!("vertex_main reads the builtin {argument} (`{text}`), which the cluster path doesn't provide")),
            "location" => location = Some(argument.parse::<u32>().map_err(|_| format!("bad location in `{text}`"))?),
            _ => {}
        }
        rest = tail.trim_start();
    }
    let (name, ty) = rest.split_once(':').ok_or_else(|| format!("no type in `{text}`"))?;
    let location = location.ok_or_else(|| format!("`{text}` has no @location"))?;
    Ok(Input { location, name: name.trim().to_string(), ty: ty.split_whitespace().collect() })
}

/// `text` without its `@attribute(...)`s.
fn strip_attributes(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text.trim();
    while let Some(after) = rest.strip_prefix('@') {
        let name_end = after.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(after.len());
        let mut tail = &after[name_end..];
        if tail.trim_start().starts_with('(') {
            let open = tail.find('(').unwrap();
            tail = &tail[matching_paren(tail, open).map_or(tail.len(), |c| c + 1)..];
        }
        rest = tail.trim_start();
    }
    out += rest;
    out
}

/// A WGSL input type's components and scalar kind ('f', 'u' or 'i').
fn shape(ty: &str) -> Option<(u32, char)> {
    let scalar = |s: &str| match s {
        "f32" | "f" => Some('f'),
        "u32" | "u" => Some('u'),
        "i32" | "i" => Some('i'),
        _ => None,
    };
    if let Some(kind) = scalar(ty) {
        return Some((1, kind));
    }
    let rest = ty.strip_prefix("vec")?;
    let n = rest.chars().next()?.to_digit(10).filter(|n| (2..=4).contains(n))?;
    let element = rest[1..].trim_start_matches('<').trim_end_matches('>');
    Some((n, scalar(element)?))
}

/// A 32-bit vertex format's components and scalar kind.
fn format_shape(format: wgpu::VertexFormat) -> Option<(u32, char)> {
    use wgpu::VertexFormat::*;
    Some(match format {
        Float32 => (1, 'f'),
        Float32x2 => (2, 'f'),
        Float32x3 => (3, 'f'),
        Float32x4 => (4, 'f'),
        Uint32 => (1, 'u'),
        Uint32x2 => (2, 'u'),
        Uint32x3 => (3, 'u'),
        Uint32x4 => (4, 'u'),
        Sint32 => (1, 'i'),
        Sint32x2 => (2, 'i'),
        Sint32x3 => (3, 'i'),
        Sint32x4 => (4, 'i'),
        _ => return None,
    })
}

/// The WGSL expression that feeds `input`: the mesh vertex's (locations 0-2, `Vertex::LAYOUT`)
/// or the instance record's attribute at its location.
fn input_expression(input: &Input, instances: Option<&InstanceBufferLayout>) -> Result<String, String> {
    let location = input.location;
    let (count, kind) = shape(&input.ty).ok_or_else(|| format!("@location({location}) {}: {} isn't a 32-bit scalar or vector", input.name, input.ty))?;
    let (expression, provided) = match location {
        0 => ("kansei_cluster_attribute(kansei_vertex, 0u, 4u)".to_string(), 'f'),
        1 => ("kansei_cluster_attribute(kansei_vertex, 4u, 3u)".to_string(), 'f'),
        2 => ("kansei_cluster_attribute(kansei_vertex, 7u, 2u)".to_string(), 'f'),
        _ => {
            let attribute = instances
                .and_then(|l| l.attributes.iter().find(|a| a.shader_location == location))
                .ok_or_else(|| format!("vertex_main reads @location({location}), which no buffer provides"))?;
            let (n, k) = format_shape(attribute.format).ok_or_else(|| format!("@location({location}): {:?} isn't a 32-bit format", attribute.format))?;
            if attribute.offset % 4 != 0 {
                return Err(format!("@location({location}): offset {} isn't a multiple of 4", attribute.offset));
            }
            let reader = match k {
                'f' => "kansei_record_f32",
                'u' => "kansei_record_u32",
                _ => "kansei_record_i32",
            };
            (format!("{reader}(kansei_record, {}u, {n}u)", attribute.offset / 4), k)
        }
    };
    if kind != provided {
        return Err(format!("@location({location}) is {} in vertex_main, but its buffer holds {}", input.ty, match provided { 'f' => "floats", 'u' => "u32s", _ => "i32s" }));
    }
    Ok(match count {
        1 => format!("{expression}.x"),
        2 => format!("{expression}.xy"),
        3 => format!("{expression}.xyz"),
        _ => expression,
    })
}
