//! Independent-validation conformance and anti-common-mode coverage.

use franken_nlp::grammar::{CompileLimits, SchemaNode, compile_json_schema};
use franken_nlp::validation::{
    Decimal, GroundedValue, JsonParseErrorKind, SourceSpan, byte_index_for_scalar, parse_json,
    scalar_index_for_byte, validate_json, validate_source_span, validate_with_grounding,
};

fn compiled_root(schema: &str) -> SchemaNode {
    compile_json_schema(schema, CompileLimits::default())
        .expect("fixture schema must compile")
        .root()
        .clone()
}

#[test]
fn policy_validation_imports_only_immutable_declarative_schema_types() {
    let sources = [
        include_str!("../src/validation/mod.rs"),
        include_str!("../src/validation/json.rs"),
        include_str!("../src/validation/schema.rs"),
        include_str!("../src/validation/offsets.rs"),
        include_str!("../src/validation/source_membership.rs"),
    ];
    let grammar_references = sources
        .iter()
        .flat_map(|source| source.lines())
        .filter(|line| line.contains("crate::grammar"))
        .collect::<Vec<_>>();
    assert_eq!(
        grammar_references,
        vec!["use crate::grammar::{ScalarValue, SchemaNode, SourceAnnotation};"],
        "validation may import only immutable declarative schema data"
    );
    for forbidden in [
        "CompiledSchema",
        "TypedJsonAutomaton",
        "CompileLimits",
        "compile_json_schema",
        "grammar::compiler",
        "grammar::execution",
        "grammar::mask",
        "grammar::source",
        "automaton()",
    ] {
        assert!(
            sources.iter().all(|source| !source.contains(forbidden)),
            "validation must not couple to grammar {forbidden}"
        );
    }
}

#[test]
fn duplicate_keys_and_hostile_json_fail_closed_with_locations() {
    let error = parse_json(r#"{"same":1,"same":2}"#).expect_err("duplicate keys must reject");
    assert_eq!(error.kind(), JsonParseErrorKind::DuplicateKey);
    assert_eq!(error.pointer(), "/same");
    assert!(error.byte_offset() > 0);

    let nesting_bomb = "[".repeat(130);
    for hostile in [
        r#"{"a":"\uD800"}"#,
        r#"{"a":"\uDC00"}"#,
        r#"{"a":"\x"}"#,
        &nesting_bomb,
        "1e309",
        "123456789012345678901234567890123456789",
    ] {
        let result = std::panic::catch_unwind(|| parse_json(hostile));
        assert!(result.is_ok(), "hostile input must not panic");
        assert!(result.expect("catch-unwind result").is_err());
    }
}

#[test]
fn decimal_comparison_is_exact_and_bounded() {
    let one = Decimal::parse("1").expect("one");
    assert_eq!(one, Decimal::parse("1.0").expect("one point zero"));
    assert_eq!(one, Decimal::parse("1e0").expect("scientific one"));
    assert_eq!(
        Decimal::parse("-0.000e12")
            .expect("zero")
            .canonical_spelling(),
        "0"
    );
    assert!(Decimal::parse("123456789012345678901234567890123456789").is_err());
    assert!(Decimal::parse("1e309").is_err());
    assert!(
        Decimal::parse("-9223372036854775808")
            .expect("i64 minimum")
            .is_integer_in_64_bit_domain()
    );
    assert!(
        Decimal::parse("18446744073709551615")
            .expect("u64 maximum")
            .is_integer_in_64_bit_domain()
    );
    assert!(
        !Decimal::parse("18446744073709551616")
            .expect("in-range decimal but not u64")
            .is_integer_in_64_bit_domain()
    );
}

#[test]
fn independent_schema_walker_accepts_exact_numeric_equivalence() {
    let schema = compiled_root(
        r#"{"type":"object","additionalProperties":false,"properties":{"n":{"type":"number","const":1.0},"i":{"type":"integer"}},"required":["n","i"]}"#,
    );
    validate_json(&schema, r#"{"n":1e0,"i":-9223372036854775808}"#)
        .expect("mathematically equal number and exact signed integer");
    let error =
        validate_json(&schema, r#"{"n":1,"i":1.5}"#).expect_err("fraction is not an exact integer");
    assert_eq!(error.pointer(), "/i");
    assert!(error.expected().contains("64-bit integer"));
    assert!(error.byte_offset().is_some());
    assert!(error.scalar_offset().is_some());
}

#[test]
fn offsets_check_utf8_boundaries_and_scalar_coordinates() {
    let source = "Aé🙂Z";
    assert_eq!(scalar_index_for_byte(source, 3), Ok(2));
    assert_eq!(byte_index_for_scalar(source, 3), Some(7));
    validate_source_span(source, "é🙂", SourceSpan::new(1, 7, 1, 3))
        .expect("multibyte half-open span");
    assert!(validate_source_span(source, "é", SourceSpan::new(2, 3, 1, 2)).is_err());
    assert!(validate_source_span(source, "é🙂", SourceSpan::new(1, 7, 0, 3)).is_err());
}

#[test]
fn verbatim_strings_require_independent_grounding_evidence() {
    let schema = compiled_root(
        r#"{"type":"object","additionalProperties":false,"properties":{"quote":{"type":"string","x-fnlp-source":"verbatim"}},"required":["quote"]}"#,
    );
    let source = "xx βeta yy";
    let grounding = GroundedValue {
        json_pointer: "/quote",
        source,
        span: SourceSpan::new(3, 8, 3, 7),
    };
    validate_with_grounding(&schema, r#"{"quote":"βeta"}"#, &[grounding])
        .expect("source span must independently prove membership");
    assert!(validate_with_grounding(&schema, r#"{"quote":"βeta"}"#, &[]).is_err());
    let wrong = GroundedValue {
        span: SourceSpan::new(3, 7, 3, 6),
        ..grounding
    };
    assert!(validate_with_grounding(&schema, r#"{"quote":"βeta"}"#, &[wrong]).is_err());
}

#[test]
fn differential_corpus_keeps_grammar_and_validation_paths_in_lockstep() {
    let compiled = compile_json_schema(
        r#"{"type":"object","additionalProperties":false,"properties":{"name":{"type":"string","maxLength":12},"count":{"type":"integer","enum":[1,2]},"items":{"type":"array","maxItems":2,"items":{"type":"boolean"}}},"required":["name","count","items"]}"#,
        CompileLimits::default(),
    )
    .expect("fixture schema");
    let schema = compiled.root().clone();
    let mut corpus = vec![
        compiled.sample_json().expect("generated valid instance"),
        r#"{"name":"ok","count":1,"items":[true,false]}"#.to_owned(),
        r#"{"name":"ok","count":1.0,"items":[]}"#.to_owned(),
        r#"{"name":"too-long-name","count":1,"items":[]}"#.to_owned(),
        r#"{"name":"ok","count":3,"items":[]}"#.to_owned(),
        r#"{"name":"ok","count":1,"items":[true,false,true]}"#.to_owned(),
        r#"{"name":"ok","name":"duplicate","count":1,"items":[]}"#.to_owned(),
        r#"{"name":"ok","count":1,"items":[],"extra":null}"#.to_owned(),
    ];
    let mut state = 0x9e37_79b9_u32;
    for _ in 0..256 {
        let mut bytes = Vec::with_capacity(96);
        for _ in 0..96 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((state >> 24) as u8);
        }
        corpus.push(String::from_utf8_lossy(&bytes).into_owned());
    }
    for input in corpus {
        let grammar = compiled.validate_json(&input).is_ok();
        let validator = validate_json(&schema, &input).is_ok();
        assert_eq!(
            grammar,
            validator,
            "paths disagree for corpus input length {}",
            input.len()
        );
        if grammar && validator {
            let reference = serde_json::from_str::<serde_json::Value>(&input);
            assert!(
                reference.is_ok(),
                "two paths accepted non-JSON reference semantics"
            );
        }
    }
}
