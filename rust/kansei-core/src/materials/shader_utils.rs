use std::collections::HashMap;

/// Map of chunk names to WGSL code snippets.
pub type ShaderChunks = HashMap<String, String>;

/// Recursively replace `#include <name>` directives with chunk content.
pub fn parse_includes(code: &str, chunks: &ShaderChunks) -> String {
    let mut result = String::new();
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#include") {
            let name = trimmed
                .trim_start_matches("#include")
                .trim()
                .trim_start_matches('<').trim_end_matches('>')
                .trim_start_matches('"').trim_end_matches('"')
                .trim();
            if let Some(chunk) = chunks.get(name) {
                result.push_str(&parse_includes(chunk, chunks));
            } else {
                // Keep the line as-is if chunk not found (will cause shader compile error)
                result.push_str(line);
                result.push('\n');
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_includes_basic() {
        let mut chunks = ShaderChunks::new();
        chunks.insert("lighting".to_string(), "fn light() -> f32 { return 1.0; }".to_string());

        let code = "// header\n#include <lighting>\n// footer";
        let result = parse_includes(code, &chunks);
        assert!(result.contains("fn light()"));
        assert!(result.contains("// header"));
        assert!(result.contains("// footer"));
    }

    #[test]
    fn test_parse_includes_recursive() {
        let mut chunks = ShaderChunks::new();
        chunks.insert("a".to_string(), "#include <b>\nfn a() {}".to_string());
        chunks.insert("b".to_string(), "fn b() {}".to_string());

        let code = "#include <a>";
        let result = parse_includes(code, &chunks);
        assert!(result.contains("fn b()"));
        assert!(result.contains("fn a()"));
    }

    #[test]
    fn test_parse_includes_missing() {
        let chunks = ShaderChunks::new();
        let code = "#include <nonexistent>";
        let result = parse_includes(code, &chunks);
        assert!(result.contains("#include <nonexistent>"));
    }
}
