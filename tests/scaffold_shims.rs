use std::process::Command;

const LONG_SHIM: &str = "fn main() -> std::process::ExitCode { franken_nlp::cli_main() }\n";
const SHORT_SHIM: &str = "fn main() -> std::process::ExitCode { franken_nlp::cli_main() }\n";

#[test]
fn both_bins_are_distinct_one_line_shims() {
    assert_eq!(include_str!("../src/main.rs"), LONG_SHIM);
    assert_eq!(include_str!("../src/bin/fnlp.rs"), SHORT_SHIM);
}

#[test]
fn both_bins_expose_the_same_help_surface() {
    let short = Command::new(env!("CARGO_BIN_EXE_fnlp"))
        .arg("--help")
        .output()
        .expect("fnlp binary must launch");
    let long = Command::new(env!("CARGO_BIN_EXE_franken_nlp"))
        .arg("--help")
        .output()
        .expect("franken_nlp binary must launch");
    assert!(short.status.success());
    assert!(long.status.success());
    assert_eq!(short.stdout, long.stdout);
}

#[test]
fn bare_invocation_prints_help_without_starting_a_tui() {
    let output = Command::new(env!("CARGO_BIN_EXE_fnlp"))
        .output()
        .expect("fnlp binary must launch");
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
}
