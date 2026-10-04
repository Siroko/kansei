//! The examples use kansei-core through its public API: none pulls in the engine's private
//! shaders or test fixtures by a relative path (`include_str!("../../../../kansei-core/src/...")`).

use std::path::Path;

fn sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() && path.file_name().is_some_and(|n| n != "target" && n != "pkg" && n != "www") {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_example_includes_engine_sources() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut files = Vec::new();
    sources(&examples, &mut files);
    assert!(files.len() > 10, "found only {} example sources under {}", files.len(), examples.display());
    let offenders: Vec<String> = files
        .iter()
        .flat_map(|file| {
            let text = std::fs::read_to_string(file).unwrap();
            text.lines()
                .enumerate()
                .filter(|(_, line)| (line.contains("include_str!") || line.contains("include_bytes!")) && line.contains("kansei-core/"))
                .map(|(i, line)| format!("{}:{}: {}", file.strip_prefix(&examples).unwrap().display(), i + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(offenders.is_empty(), "examples reaching into kansei-core's files (use its public constants instead):\n{}", offenders.join("\n"));
}
