//! Model-free conformance coverage for the bounded v1 schema compiler.

use franken_nlp::grammar::{CompileLimits, ExactDecimal, SchemaError, compile_json_schema};

fn compile(source: &str) -> franken_nlp::grammar::CompiledSchema {
    compile_json_schema(source, CompileLimits::default()).expect("fixture schema must compile")
}

#[test]
fn schema_sampling_and_native_extraction_agree_on_unicode_lengths() {
    use franken_nlp::grammar::runtime::JsonProgram;
    let limits = CompileLimits { max_string_bytes: 4, ..CompileLimits::default() };
    let schema = r#"{"type":"object","additionalProperties":false,"required":["label"],"properties":{"label":{"type":"string","maxLength":1,"const":"🐈"}}}"#;
    let compiled = compile_json_schema(schema, limits).unwrap();
    let sample = compiled.sample_json().unwrap();
    assert_eq!(sample, r#"{"label":"🐈"}"#);
    JsonProgram::compile(schema, limits).unwrap().validate_json(&sample).unwrap();

    let schema = r#"{"type":"array","items":{"type":"string","maxLength":1}}"#;
    let compiled = compile_json_schema(schema, limits).unwrap();
    let native = JsonProgram::compile(schema, limits).unwrap();
    for (instance, valid) in [
        (r#"["é","🐈"]"#, true),
        (r#"["\uD83D\uDC08"]"#, true),
        (r#"["e\u0301"]"#, false),
        (r#"["ab"]"#, false),
    ] {
        assert_eq!(compiled.validate_json(instance).is_ok(), valid);
        assert_eq!(native.validate_json(instance).is_ok(), valid);
    }
    let narrow = CompileLimits { max_string_bytes: 3, ..limits };
    assert!(compile_json_schema(schema, narrow).unwrap().validate_json(r#"["🐈"]"#).is_err());
    assert!(JsonProgram::compile(schema, narrow).unwrap().validate_json(r#"["🐈"]"#).is_err());
}

#[test]
fn supported_nested_schema_samples_and_revalidates_without_a_model() {
    let compiled = compile(
        r#"{
          "type":"object",
          "additionalProperties":false,
          "properties":{
            "active":{"type":"boolean"},
            "count":{"type":"integer","enum":[1,2]},
            "label":{"type":"string","maxLength":12,"x-fnlp-source":"verbatim"},
            "ratio":{"type":"number","const":1.0},
            "tags":{"type":"array","maxItems":3,"items":{"type":"string"}},
            "nothing":{"type":"null"}
          },
          "required":["active","count","label","ratio","tags","nothing"]
        }"#,
    );
    let sample = compiled.sample_json().expect("sample must be valid");
    compiled
        .validate_json(&sample)
        .expect("sample must revalidate through the same exact parser");
    assert!(compiled.requires_verbatim_source());
    assert!(compiled.automaton().every_state_reaches_acceptance());
    assert!(compiled.estimate().state_count > 0);
    assert!(compiled.estimate().mask_cache_bytes > 0);

    compiled
        .validate_json(
            r#"{"active":false,"count":2,"label":"ok","ratio":1e0,"tags":["a","b"],"nothing":null}"#,
        )
        .expect("mathematically equal numeric const must validate");
}

#[test]
fn exact_decimal_equality_and_canonicalization_never_use_float_rounding() {
    let one = ExactDecimal::parse("1").expect("valid decimal");
    assert_eq!(one, ExactDecimal::parse("1.0").expect("valid decimal"));
    assert_eq!(one, ExactDecimal::parse("1e0").expect("valid decimal"));
    assert_eq!(
        ExactDecimal::parse("-0.000e12").expect("zero"),
        ExactDecimal::parse("0").expect("zero")
    );
    assert_eq!(
        ExactDecimal::parse("-0.000e12")
            .expect("zero")
            .canonical_spelling(),
        "0"
    );
    assert_eq!(
        ExactDecimal::parse("12.30")
            .expect("decimal")
            .canonical_spelling(),
        "1.23e1"
    );
    assert_eq!(
        ExactDecimal::parse("-9223372036854775808")
            .expect("i64 minimum")
            .canonical_spelling(),
        "-9223372036854775808"
    );
    assert!(ExactDecimal::parse("123456789012345678901234567890123456789").is_err());
    assert!(ExactDecimal::parse("1e309").is_err());
}

#[test]
fn mathematically_duplicate_enum_members_reject_before_automaton_build() {
    let error = compile_json_schema(
        r#"{"type":"number","enum":[1,1.0,1e0]}"#,
        CompileLimits::default(),
    )
    .expect_err("mathematically equal enum values must not fork generation");
    assert!(
        error
            .to_string()
            .contains("duplicate mathematically equal scalar")
    );
    assert_eq!(error.pointer(), "/enum/1");
}

#[test]
fn rejection_by_keyword_is_typed_and_pointer_located() {
    for keyword in [
        "$ref",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "patternProperties",
        "uniqueItems",
        "contains",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
        "pattern",
        "verbatim-normalized",
    ] {
        let schema = format!(r#"{{"type":"string","{keyword}":true}}"#);
        let error = compile_json_schema(&schema, CompileLimits::default())
            .expect_err("unsupported keyword must reject before model load");
        assert_eq!(error.keyword(), Some(keyword));
        assert_eq!(error.pointer(), format!("/{keyword}").as_str());
    }
    let error = compile_json_schema(
        r#"{"type":"object","additionalProperties":true}"#,
        CompileLimits::default(),
    )
    .expect_err("open objects are outside v1");
    assert_eq!(error.keyword(), Some("additionalProperties:true"));
}

#[test]
fn duplicate_keys_and_resource_caps_fail_before_automaton_allocation() {
    let duplicate = compile_json_schema(
        r#"{"type":"string","type":"string"}"#,
        CompileLimits::default(),
    )
    .expect_err("duplicate schema keys must not overwrite");
    assert!(matches!(duplicate, SchemaError::DuplicateKey { .. }));

    let limits = CompileLimits {
        max_states: 1,
        ..CompileLimits::default()
    };
    let error = compile_json_schema(
        r#"{"type":"object","additionalProperties":false,"properties":{"x":{"type":"string"}},"required":["x"]}"#,
        limits,
    )
    .expect_err("state cap must reject before automaton allocation");
    assert!(matches!(error, SchemaError::Resource { .. }));
    assert!(error.to_string().contains("before allocation"));
}

#[test]
fn validation_rejects_unknown_properties_nonintegers_and_overlong_arrays() {
    let compiled = compile(
        r#"{
          "type":"object",
          "additionalProperties":false,
          "properties":{
            "count":{"type":"integer"},
            "entries":{"type":"array","maxItems":1,"items":{"type":"null"}}
          },
          "required":["count","entries"]
        }"#,
    );
    assert!(
        compiled
            .validate_json(r#"{"count":1.5,"entries":[]}"#)
            .is_err()
    );
    assert!(
        compiled
            .validate_json(r#"{"count":1,"entries":[null,null]}"#)
            .is_err()
    );
    let error = compiled
        .validate_json(r#"{"count":1,"entries":[],"surplus":true}"#)
        .expect_err("additionalProperties:false is an instance boundary");
    assert_eq!(error.pointer(), "/surplus");
}

#[test]
fn arbitrary_byte_like_input_never_panics_the_compiler() {
    let mut state = 0x6d2b_79f5_u32;
    for _ in 0..256 {
        let mut bytes = Vec::with_capacity(64);
        for _ in 0..64 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((state >> 24) as u8);
        }
        let input = String::from_utf8_lossy(&bytes);
        let result =
            std::panic::catch_unwind(|| compile_json_schema(&input, CompileLimits::default()));
        assert!(result.is_ok(), "compiler panicked for fuzz seed");
    }
}

#[test]
fn source_bound_schema_commands_compile_and_sample_the_document_language() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    fn invoke(args: &[&str], input: &[u8]) -> std::process::Output {
        let mut child = Command::new(env!("CARGO_BIN_EXE_fnlp"))
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn().unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        child.wait_with_output().unwrap()
    }

    let schema = "tests/fixtures/schema/accept-grounded-choice.schema.json";
    let document = "Private source with 東京.".as_bytes();
    let checked = invoke(&["schema", "check", schema, "--source", "-"], document);
    assert!(checked.status.success());
    assert!(checked.stdout.is_empty());
    assert!(String::from_utf8_lossy(&checked.stderr).contains("source_product=BOUND"));
    let sample = invoke(&["schema", "sample", schema, "--source", "-"], document);
    assert!(sample.status.success());
    assert_eq!(String::from_utf8(sample.stdout).unwrap(), "{\"city\":\"東京\"}\n");
    assert!(!String::from_utf8_lossy(&sample.stderr).contains("東京"));

    for mode in ["check", "sample"] {
        let missing = invoke(&["schema", mode, schema, "--source", "-"], b"elsewhere");
        assert_eq!(missing.status.code(), Some(8));
        assert!(missing.stdout.is_empty());
        assert!(!String::from_utf8_lossy(&missing.stderr).contains("elsewhere"));
    }

    let from_file = invoke(&["schema", "sample", "-", "--source", "Cargo.toml"],
        br#"{"type":"string","enum":["[package]","missing-value"],"x-fnlp-source":"verbatim"}"#);
    assert!(from_file.status.success());
    assert_eq!(from_file.stdout, b"\"[package]\"\n");

    let duplicate_stdin = invoke(&["schema", "check", "-", "--source", "-"], b"");
    assert_eq!(duplicate_stdin.status.code(), Some(2));
    let invalid = invoke(&["schema", "check", schema, "--source", "-"], &[0xff]);
    assert_eq!(invalid.status.code(), Some(4));
    let oversized = invoke(&["schema", "sample", schema, "--source", "-"], &vec![b'A'; 1024 * 1024 + 1]);
    assert_eq!(oversized.status.code(), Some(9));
    assert!(oversized.stdout.is_empty());
}

#[test]
fn schema_sampling_reserves_the_minimum_for_every_required_sibling() {
    let schema = r#"{"type":"object","additionalProperties":false,"required":["a","b","c"],"properties":{"a":{"type":"boolean"},"b":{"type":"string","enum":["aaaa","z"]},"c":{"type":"boolean","enum":[false,true]}}}"#;
    let minimum = r#"{"a":true,"b":"z","c":true}"#;
    let preferred = r#"{"a":false,"b":"aaaa","c":false}"#;
    for cap in minimum.len()..=preferred.len() {
        let limits = CompileLimits { max_output_bytes: cap, ..CompileLimits::default() };
        let compiled = compile_json_schema(schema, limits).unwrap();
        assert_eq!(compiled.estimate().minimum_output_bytes, minimum.len());
        let sample = compiled.sample_json().unwrap();
        assert!(sample.len() <= cap);
        compiled.validate_json(&sample).unwrap();
        if cap == minimum.len() { assert_eq!(sample, minimum); }
        if cap == preferred.len() { assert_eq!(sample, preferred); }
    }
}

#[test]
fn resource_caps_include_the_shared_accept_state_and_root_transition() {
    let schema = r#"{"type":"string"}"#;
    let compiled = compile(schema);
    let estimate = compiled.estimate();
    assert_eq!(estimate.state_count, compiled.automaton().state_count());
    assert_eq!(estimate.transition_count, compiled.automaton().transition_count());
    for limits in [
        CompileLimits { max_states: estimate.state_count - 1, ..CompileLimits::default() },
        CompileLimits { max_transitions: estimate.transition_count - 1, ..CompileLimits::default() },
        CompileLimits { max_mask_bytes: estimate.mask_cache_bytes - 1, ..CompileLimits::default() },
    ] {
        assert!(matches!(compile_json_schema(schema, limits), Err(SchemaError::Resource { .. })));
    }
}
