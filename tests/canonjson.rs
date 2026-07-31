#[path = "../src/canonjson.rs"]
mod canonjson;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use canonjson::{
    CanonJsonError, ParseLimits, canonical_bytes, canonicalize_str, parse_str,
    parse_str_with_limits,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct NestedFixture {
    b: String,
    a: String,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct WriterFixture {
    z: u64,
    escaped: String,
    list: Vec<Option<i64>>,
    nested: NestedFixture,
    flag: bool,
    a: BTreeMap<String, String>,
}

#[test]
fn duplicate_keys_reject_at_exact_json_pointer_paths() {
    assert_duplicate_path(r#"{"alpha":1,"alpha":2}"#, "/alpha");
    assert_duplicate_path(r#"{"outer":{"same":1,"same":2}}"#, "/outer/same");
    assert_duplicate_path(r#"{"items":[{"name":1,"name":2}]}"#, "/items/0/name");
    assert_duplicate_path(r#"{"a/b":{"~key":1,"~key":2}}"#, "/a~1b/~0key");
}

#[test]
fn valid_input_matches_serde_json_value_semantics() {
    let input = r#"{"object":{"z":1,"a":"two"},"array":[true,null,3],"text":"é"}"#;
    let rejecting = parse_str(input).expect("valid JSON must parse");
    let serde_value: Value = serde_json::from_str(input).expect("serde must parse valid JSON");
    assert_eq!(rejecting, serde_value);
}

#[test]
fn parse_limits_are_typed_and_path_bearing() {
    let depth_error = parse_str_with_limits(
        r#"{"outer":{"inner":1}}"#,
        ParseLimits {
            max_depth: 1,
            max_string_bytes: 16,
        },
    )
    .expect_err("nested object must exceed depth cap");
    assert!(matches!(
        depth_error,
        CanonJsonError::DepthLimit { ref path, .. } if path.to_string() == "/outer"
    ));
    assert!(depth_error.to_string().contains("/outer"));

    let string_error = parse_str_with_limits(
        r#"{"value":"four"}"#,
        ParseLimits {
            max_depth: 4,
            max_string_bytes: 3,
        },
    )
    .expect_err("string must exceed byte cap");
    assert!(matches!(
        string_error,
        CanonJsonError::StringLimit { ref path, .. } if path.to_string() == "/value"
    ));
    assert!(string_error.to_string().contains("/value"));
}

#[test]
fn canonical_writer_matches_pinned_byte_golden_and_rejects_nonfinite() {
    let mut map = BTreeMap::new();
    map.insert("z".to_owned(), "last".to_owned());
    map.insert("a".to_owned(), "first".to_owned());
    let fixture = WriterFixture {
        z: 9,
        escaped: "\"\\\u{0008}\t\n\u{000C}\r\u{0001}é".to_owned(),
        list: vec![None, Some(-7), Some(9)],
        nested: NestedFixture {
            b: "B".to_owned(),
            a: "A".to_owned(),
        },
        flag: true,
        a: map,
    };

    let actual = canonical_bytes(&fixture).expect("finite fixture must serialize");
    let expected = concat!(
        r#"{"a":{"a":"first","z":"last"},"escaped":"\"\\\b\t\n\f\r\u0001"#,
        "é",
        r#"","flag":true,"list":[null,-7,9],"nested":{"a":"A","b":"B"},"z":9}"#
    )
    .as_bytes();
    assert_eq!(actual, expected);
    assert_eq!(
        canonicalize_str(r#"{"z":9,"a":1}"#, ParseLimits::default()).unwrap(),
        br#"{"a":1,"z":9}"#
    );

    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            canonical_bytes(&value),
            Err(CanonJsonError::NonFiniteNumber)
        ));
    }
}

#[test]
fn deterministic_property_samples_round_trip_and_duplicate_injections_reject() {
    for seed in 0_u64..256 {
        let value = WriterFixture {
            z: seed,
            escaped: format!("seed={seed}; escape=\\\"\n; unicode=é"),
            list: vec![Some(seed as i64), None, Some(-(seed as i64))],
            nested: NestedFixture {
                b: format!("b-{seed}"),
                a: format!("a-{}", seed ^ 0x5a),
            },
            flag: seed % 2 == 0,
            a: BTreeMap::from([
                (format!("z-{seed}"), "last".to_owned()),
                (format!("a-{seed}"), "first".to_owned()),
            ]),
        };
        let bytes = canonical_bytes(&value).expect("finite fixture must serialize");
        let text = std::str::from_utf8(&bytes).expect("canonical JSON must be UTF-8");
        let reparsed = parse_str(text).expect("writer output must pass rejecting parser");
        let typed: WriterFixture = serde_json::from_value(reparsed).expect("typed round trip");
        assert_eq!(typed, value);

        let injected = format!(r#"{{"dup":{seed},"dup":{}}}"#, text);
        let error = parse_str(&injected).expect_err("duplicate injection must reject");
        assert!(matches!(error, CanonJsonError::DuplicateKey { .. }));
        assert!(error.to_string().contains("/dup"));
    }
}

#[test]
fn raw_json_parse_policy_is_confined_to_the_canonjson_chokepoint() {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&source_root, &mut rust_sources);

    for source in rust_sources {
        if source
            .file_name()
            .is_some_and(|name| name == "canonjson.rs")
        {
            continue;
        }
        let text = fs::read_to_string(&source).expect("source file must be readable");
        for forbidden in ["serde_json::from_str", "serde_json::from_reader"] {
            assert!(
                !text.contains(forbidden),
                "raw parser {forbidden} outside canonjson chokepoint: {}",
                source.display()
            );
        }
    }
}

#[test]
fn canonjson_contract_logs_pass_marker() {
    eprintln!("CANONJSON RESULT=PASS");
}

fn assert_duplicate_path(input: &str, expected_path: &str) {
    let error = parse_str(input).expect_err("duplicate keys must reject");
    assert!(matches!(
        error,
        CanonJsonError::DuplicateKey { ref path } if path.to_string() == expected_path
    ));
    assert!(error.to_string().contains(expected_path));
}

fn collect_rust_sources(directory: &Path, output: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(directory).expect("source directory must be readable") {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            collect_rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}
