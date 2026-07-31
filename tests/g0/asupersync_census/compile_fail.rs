//! OQ-35 census — compiler-enforced least-authority boundaries.
//!
//! This deliberately uses the pinned toolchain directly rather than adding a
//! release-graph dependency for a UI-test helper. Each fixture is compiled
//! against the exact `asupersync` rlib that built this feature-gated target.
//! This target proves only a static narrowing boundary. It does not establish
//! that ambient `Cx::current()` enforces reduced effects: the pin returns
//! `Cx<cap::All>` there, so product leaves must receive an explicit narrowed
//! context instead.

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

struct CompileFailCase {
    source: &'static str,
    required_diagnostic: &'static str,
}

const CASES: [CompileFailCase; 2] = [
    CompileFailCase {
        source: "tests/g0/compile_fail/capability_widening.rs",
        required_diagnostic: "SubsetOf",
    },
    CompileFailCase {
        source: "tests/g0/compile_fail/cx_current_regain.rs",
        required_diagnostic: "no function or associated item named",
    },
];

fn dependency_dir() -> PathBuf {
    env::current_exe()
        .expect("census test executable path is available")
        .parent()
        .expect("census test executable has a dependency directory")
        .to_path_buf()
}

fn asupersync_rlib(dependency_dir: &Path) -> PathBuf {
    let mut candidates = fs::read_dir(dependency_dir)
        .expect("census dependency directory is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "rlib")
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("libasupersync-"))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .pop()
        .expect("feature-gated census target links one asupersync rlib")
}

fn compile_failure(case: &CompileFailCase, dependency_dir: &Path, asupersync: &Path) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join(case.source);
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let extern_argument = format!("asupersync={}", asupersync.display());
    let output = Command::new(rustc)
        .arg("--edition=2024")
        .arg("--crate-type=bin")
        .arg("--emit=metadata")
        .arg("--error-format=short")
        .arg("--out-dir")
        .arg(dependency_dir)
        .arg("-L")
        .arg(format!("dependency={}", dependency_dir.display()))
        .arg("--extern")
        .arg(extern_argument)
        .arg(&source)
        .output()
        .expect("pinned rustc starts the compile-fail fixture");
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "{} unexpectedly compiled; OQ-35 least-authority boundary regressed",
        case.source
    );
    assert!(
        diagnostics.contains(case.required_diagnostic),
        "{} failed for the wrong reason; expected diagnostic containing {:?}, got:\n{}",
        case.source,
        case.required_diagnostic,
        diagnostics
    );
    diagnostics
}

#[test]
fn static_widening_and_no_generic_restricted_current_api_fail_to_compile() {
    let dependency_dir = dependency_dir();
    let asupersync = asupersync_rlib(&dependency_dir);

    for case in &CASES {
        let diagnostics = compile_failure(case, &dependency_dir, &asupersync);
        println!(
            "G0_CENSUS item=compile-fail-suite case={} result=expected-failure diagnostic={}",
            case.source, case.required_diagnostic,
        );
        assert!(
            !diagnostics.is_empty(),
            "a rejected fixture must retain compiler diagnostics"
        );
    }

    println!(
        "G0_CENSUS item=compile-fail-suite RESULT=RATIFIED evidence=restricted-to-all-rejected+generic-restricted-current-api-absent;no-ambient-authority-claim"
    );
}
