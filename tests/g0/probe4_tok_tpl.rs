//! G0-04 fixture-backed tokenizer and chat-template spike.
//!
//! This is a narrow corpus-integrity probe, not a second tokenizer or template
//! implementation. It locks the Phase -1 evidence shape which the L0
//! implementations must consume. `ADR-G0-04-tokenizer-template.md` remains
//! BLOCKED until its implementation-level differential runs are recorded.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

const REQUIRED_TEMPLATE_CASES: &[&str] = [
    "system-default-no-think",
    "thinking-preserved",
    "tool-json",
    "tool-xml",
    "media-reminder",
];
const REQUIRED_TOKENIZER_CASES: &[&str] = [
    "ascii-whitespace",
    "multilingual",
    "code-punctuation",
    "marker-literals",
];

fn fixture_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference")
}

fn parse_object(path: &Path) -> Value {
    let bytes = fs::read(path).expect("reference fixture must remain readable");
    serde_json::from_slice(&bytes).expect("reference fixture must remain JSON")
}

fn records_by_id(value: &Value, key: &str) -> BTreeSet<String> {
    value[key]
        .as_array()
        .expect("reference fixture record array")
        .iter()
        .map(|record| {
            record["id"]
                .as_str()
                .expect("reference fixture record id")
                .to_owned()
        })
        .collect()
}

#[test]
fn phase_minus_one_auxiliary_fixture_covers_l0_tokenizer_and_template_cases() {
    let auxiliary = parse_object(&fixture_root().join("auxiliary.json"));
    let tokenizer_cases = records_by_id(&auxiliary, "tokenizer_cases");
    let template_cases = records_by_id(&auxiliary, "template_cases");

    for required in REQUIRED_TOKENIZER_CASES {
        assert!(
            tokenizer_cases.contains(*required),
            "missing frozen slow-tokenizer case={required}"
        );
    }
    for required in REQUIRED_TEMPLATE_CASES {
        assert!(
            template_cases.contains(*required),
            "missing frozen chat-template case={required}"
        );
    }

    for record in auxiliary["tokenizer_cases"]
        .as_array()
        .expect("tokenizer cases array")
    {
        let ids = record["token_ids"]
            .as_array()
            .expect("slow-tokenizer ids must be an array");
        assert!(
            !ids.is_empty(),
            "tokenizer fixture must not silently accept an empty id sequence"
        );
        assert!(
            record["token_ids_sha256"].as_str().is_some(),
            "tokenizer fixture must bind its id sequence by digest"
        );
    }

    for record in auxiliary["template_cases"]
        .as_array()
        .expect("template cases array")
    {
        assert!(
            record["rendered_sha256"].as_str().is_some(),
            "template fixture must bind rendered bytes by digest"
        );
        assert!(
            record["token_ids_sha256"].as_str().is_some(),
            "template fixture must bind rendered token ids by digest"
        );
    }

    println!(
        "G0_PROBE4 case=frozen-l0-corpus RESULT=PASS tokenizer_cases={} template_cases={} authority=fixture-contract-only",
        tokenizer_cases.len(),
        template_cases.len()
    );
    println!("G0_PROBE4 RESULT=PASS cases=1 authority=fixture-contract-only");
}
