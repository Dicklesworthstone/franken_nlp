//! Source-level enforcement for the deny-by-default unsafe policy.
//!
//! This deliberately uses only `std`: the policy must guard the dependency
//! graph itself, so it cannot rely on a parser crate that changes that graph.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const ALLOWLIST: &str = "tests/fixtures/unsafe_policy/island_allowlist.txt";
const CRATE_ROOTS: [&str; 3] = ["src/lib.rs", "src/main.rs", "src/bin/fnlp.rs"];

#[derive(Debug)]
struct Violation {
    path: String,
    line: usize,
    lint: &'static str,
    reason: String,
}

impl Violation {
    fn new(path: impl Into<String>, line: usize, lint: &'static str, reason: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line,
            lint,
            reason: reason.into(),
        }
    }
}

#[test]
fn current_tree_obeys_the_unsafe_policy() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let islands = read_allowlist(root);
    let mut violations = Vec::new();

    check_cargo_lints(root, &mut violations);
    check_crate_roots(root, &mut violations);

    let mut source_files = Vec::new();
    collect_rust_sources(&root.join("src"), &mut source_files);
    source_files.sort();

    println!("UNSAFE_POLICY islands={}", islands.len());
    for island in &islands {
        println!("UNSAFE_POLICY island={island}");
    }

    for path in source_files {
        let relative = relative_path(root, &path);
        let source = read_text(&path);
        scan_source(&relative, &source, islands.contains(&relative), &mut violations);
    }

    report(violations);
}

#[test]
fn negative_fixtures_are_rejected_and_a_safe_island_fixture_is_accepted() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture_root = root.join("tests/fixtures/unsafe_policy");
    let no_islands = BTreeSet::new();

    let mut unlisted = Vec::new();
    scan_source(
        "fixture/unlisted_allow.rs",
        &read_text(&fixture_root.join("unlisted_allow.rs")),
        no_islands.contains("fixture/unlisted_allow.rs"),
        &mut unlisted,
    );
    assert!(
        unlisted.iter().any(|violation| {
            violation.lint == "unsafe_code" && violation.reason.contains("unlisted island")
        }),
        "unlisted fixture did not name its unsafe_code violation: {unlisted:#?}"
    );

    let mut unsafe_op = Vec::new();
    scan_source(
        "fixture/unsafe_op_allow.rs",
        &read_text(&fixture_root.join("unsafe_op_allow.rs")),
        false,
        &mut unsafe_op,
    );
    assert!(
        unsafe_op
            .iter()
            .any(|violation| violation.lint == "unsafe_op_in_unsafe_fn"),
        "unsafe_op_in_unsafe_fn allow was not rejected: {unsafe_op:#?}"
    );

    let mut missing_safety = Vec::new();
    scan_source(
        "fixture/allowed_missing_safety.rs",
        &read_text(&fixture_root.join("allowed_missing_safety.rs")),
        true,
        &mut missing_safety,
    );
    assert!(
        missing_safety.iter().any(|violation| {
            violation.lint == "unsafe_code" && violation.reason.contains("SAFETY")
        }),
        "island fixture without a SAFETY proof was not rejected: {missing_safety:#?}"
    );

    let mut safe = Vec::new();
    scan_source(
        "fixture/allowed_with_safety.rs",
        &read_text(&fixture_root.join("allowed_with_safety.rs")),
        true,
        &mut safe,
    );
    assert!(safe.is_empty(), "safe island fixture was rejected: {safe:#?}");
}

fn read_allowlist(root: &Path) -> BTreeSet<String> {
    read_text(&root.join(ALLOWLIST))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect()
}

fn check_cargo_lints(root: &Path, violations: &mut Vec<Violation>) {
    let manifest = read_text(&root.join("Cargo.toml"));
    let section = lints_rust_section(&manifest);
    for lint in ["unsafe_code", "unsafe_op_in_unsafe_fn"] {
        let expected = format!("{lint} = \"deny\"");
        if !section.lines().any(|line| line.trim() == expected) {
            violations.push(Violation::new(
                "Cargo.toml",
                line_number(&manifest, "[lints.rust]"),
                lint,
                format!("[lints.rust] must contain {expected}"),
            ));
        }
    }
}

fn lints_rust_section(manifest: &str) -> &str {
    let Some(start) = manifest.find("[lints.rust]") else {
        return "";
    };
    let after_header = &manifest[start + "[lints.rust]".len()..];
    let end = after_header
        .find("\n[")
        .unwrap_or(after_header.len());
    &after_header[..end]
}

fn check_crate_roots(root: &Path, violations: &mut Vec<Violation>) {
    for relative in CRATE_ROOTS {
        let path = root.join(relative);
        let source = read_text(&path);
        if !source
            .lines()
            .any(|line| line.trim() == "#![deny(unsafe_code)]")
        {
            violations.push(Violation::new(
                relative,
                1,
                "unsafe_code",
                "crate root must contain #![deny(unsafe_code)]",
            ));
        }
    }
}

fn collect_rust_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("source directory must be readable") {
        let entry = entry.expect("source directory entry must be readable");
        let file_type = entry.file_type().expect("source entry type must be readable");
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            collect_rust_sources(&path, sources);
        } else if file_type.is_file() && path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

fn scan_source(path: &str, source: &str, is_island: bool, violations: &mut Vec<Violation>) {
    for (line, attribute) in rust_attributes(source) {
        check_governed_lints(path, line, &attribute, is_island, violations);
    }

    let lines: Vec<&str> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        if !starts_unsafe_block(line, lines.get(index + 1).copied()) {
            continue;
        }
        let line_number = index + 1;
        if !is_island {
            violations.push(Violation::new(
                path,
                line_number,
                "unsafe_code",
                "unsafe block is outside an allowlisted island",
            ));
        } else if !has_adjacent_safety_comment(&lines, index) {
            violations.push(Violation::new(
                path,
                line_number,
                "unsafe_code",
                "unsafe block in an island lacks an adjacent // SAFETY: proof",
            ));
        }
    }
}

fn rust_attributes(source: &str) -> Vec<(usize, String)> {
    let mut attributes = Vec::new();
    let mut pending: Option<(usize, String)> = None;

    for (index, original_line) in source.lines().enumerate() {
        let line = original_line.split("//").next().unwrap_or_default();
        if let Some((start, text)) = pending.as_mut() {
            text.push('\n');
            text.push_str(line);
            if line.contains(']') {
                attributes.push((*start, text.clone()));
                pending = None;
            }
            continue;
        }

        let ordinary = line.find("#[");
        let inner = line.find("#![");
        let start_column = match (ordinary, inner) {
            (Some(left), Some(right)) => left.min(right),
            (Some(column), None) | (None, Some(column)) => column,
            (None, None) => continue,
        };
        let attribute = line[start_column..].to_owned();
        if attribute.contains(']') {
            attributes.push((index + 1, attribute));
        } else {
            pending = Some((index + 1, attribute));
        }
    }
    attributes
}

fn check_governed_lints(
    path: &str,
    line: usize,
    attribute: &str,
    is_island: bool,
    violations: &mut Vec<Violation>,
) {
    let compact: String = attribute.chars().filter(|character| !character.is_whitespace()).collect();
    let level = ["allow", "warn", "expect", "forbid", "deny"]
        .into_iter()
        .find(|level| compact.contains(&format!("{level}(")));
    let Some(level) = level else {
        return;
    };

    if compact.contains("unsafe_op_in_unsafe_fn") && level != "deny" {
        violations.push(Violation::new(
            path,
            line,
            "unsafe_op_in_unsafe_fn",
            format!("{level} is forbidden; every unsafe operation needs an explicit unsafe block"),
        ));
    }

    if compact.contains("unsafe_code") {
        let allowed_island = level == "allow" && is_island;
        if !allowed_island && level != "deny" {
            let reason = if level == "allow" {
                "allow(unsafe_code) appears in an unlisted island module".to_owned()
            } else {
                format!("{level}(unsafe_code) weakens the required deny-not-forbid policy")
            };
            violations.push(Violation::new(path, line, "unsafe_code", reason));
        }
    }
}

fn starts_unsafe_block(line: &str, next_line: Option<&str>) -> bool {
    let code = line.split("//").next().unwrap_or_default().trim();
    code.contains("unsafe {")
        || (code.ends_with("unsafe")
            && next_line
                .map(|next| next.split("//").next().unwrap_or_default().trim().starts_with('{'))
                .unwrap_or(false))
}

fn has_adjacent_safety_comment(lines: &[&str], index: usize) -> bool {
    lines[index].contains("// SAFETY:")
        || index > 0 && lines[index - 1].trim_start().starts_with("// SAFETY:")
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("source path must be below manifest root")
        .to_string_lossy()
        .replace('\\', "/")
}

fn line_number(source: &str, needle: &str) -> usize {
    source
        .lines()
        .position(|line| line.contains(needle))
        .map(|index| index + 1)
        .unwrap_or(1)
}

fn read_text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

fn report(violations: Vec<Violation>) {
    if violations.is_empty() {
        println!("UNSAFE_POLICY RESULT=PASS violations=0");
        return;
    }

    for violation in &violations {
        println!(
            "{}:{}: {} {}",
            violation.path, violation.line, violation.lint, violation.reason
        );
    }
    println!("UNSAFE_POLICY RESULT=FAIL violations={}", violations.len());
    panic!("unsafe policy violations: {}", violations.len());
}
