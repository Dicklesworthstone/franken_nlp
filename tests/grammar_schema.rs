//! Model-free conformance coverage for the bounded v1 schema compiler.

use franken_nlp::grammar::{CompileLimits, ExactDecimal, SchemaError, compile_json_schema};

fn compile(source: &str) -> franken_nlp::grammar::CompiledSchema {
    compile_json_schema(source, CompileLimits::default()).expect("fixture schema must compile")
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
