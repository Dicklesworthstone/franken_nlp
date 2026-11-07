//! Definition-level regressions; execution/qualification remains controller-owned.
use super::*;
use crate::{grammar::{CompileLimits, SourceAnnotation, runtime::JsonValidationSchema},
    validation::grounded_fields::SourceOccurrence};

#[derive(Default)]
struct Control { polls: usize, stop: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    fn prefill_checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.stop == Some(self.polls)).then_some(DecodeCancellationKind::Deadline)
    }
}
fn limits() -> RootArrayUnionLimits {
    RootArrayUnionLimits { max_records: 64, max_origins: 128, max_input_bytes: 65536,
        max_json_bytes: 65536, max_nodes: 4096, max_result_bytes: 131072 }
}
fn numbers(cap: usize) -> SchemaNode {
    SchemaNode::Array { items: Box::new(SchemaNode::Number { integer: false, allowed: None }), max_items: cap }
}
fn span(start: usize, end: usize) -> VerifiedSourceSpan {
    VerifiedSourceSpan { byte_start: start, byte_end: end, scalar_start: start, scalar_end: end }
}
fn run(schema: &SchemaNode, inputs: &[&str], limits: RootArrayUnionLimits, control: &mut Control)
    -> Result<RootArrayUnionResult, RootArrayUnionError> {
    reduce(schema, Sha256Digest::of_bytes(b"sealed fixture snapshot"), limits,
        inputs.iter().enumerate().map(|(i, json)| ArrayUnionChunk { chunk_id: i,
            source_span: span(i, i + 1), json, source_fields: &[] }),
        |value| validation::validate_value(schema, value).map_err(|_| RootArrayUnionError::Input), control)
}

#[test]
fn exact_numbers_deduplicate_without_f64_and_keep_first_occurrence_order() {
    let output = run(&numbers(64), &["[9007199254740992,9007199254740993,1.0,-0.0]",
        "[1e0,0,9007199254740992]"], limits(), &mut Control::default()).unwrap();
    assert_eq!(output.json(), "[9007199254740992,9007199254740993,1,0]");
    assert_eq!(output.input_records(), 7); assert_eq!(output.duplicate_records(), 3);
    assert_eq!(output.records()[0].origins.iter().map(|o| (o.chunk_id, o.array_index)).collect::<Vec<_>>(),
        [(0, 0), (1, 2)]);
    assert_eq!(output.records()[2].origins.iter().map(|o| (o.chunk_id, o.array_index)).collect::<Vec<_>>(),
        [(0, 2), (1, 0)]);
}
#[test]
fn thirty_eight_digit_decimals_are_canonical_and_distinct() {
    let output = run(&numbers(64), &["[12345678901234567890123456789012345678,12345678901234567890123456789012345679]"],
        limits(), &mut Control::default()).unwrap();
    assert_eq!(output.json(), "[1.2345678901234567890123456789012345678e37,1.2345678901234567890123456789012345679e37]");
    assert_eq!(output.records().len(), 2);
}
#[test]
fn every_duplicate_origin_references_its_complete_original_evidence() {
    let schema = JsonValidationSchema::compile(r#"{"type":"array","maxItems":8,"items":{"type":"object","properties":{"a/~":{"type":"string","x-fnlp-source":"verbatim"}},"required":["a/~"],"additionalProperties":false}}"#,
        CompileLimits::default()).unwrap();
    let first = vec![SourceFieldEvidence { json_pointer: "/0/a~1~0".to_owned(), occurrence: SourceOccurrence::Ambiguous,
        spans: vec![span(0, 1), span(2, 3)] }];
    let second = vec![SourceFieldEvidence { json_pointer: "/0/a~1~0".to_owned(), occurrence: SourceOccurrence::Anchored,
        spans: vec![span(4, 5)] }];
    let inputs = [ArrayUnionChunk { chunk_id: 0, source_span: span(0, 3), json: r#"[{"a/~":"A"}]"#, source_fields: &first },
        ArrayUnionChunk { chunk_id: 1, source_span: span(3, 6), json: r#"[{"a/~":"\u0041"}]"#, source_fields: &second }];
    let output = reduce(schema.root(), Sha256Digest::of_bytes(b"fixture"), limits(), inputs.into_iter(),
        |value| schema.validate_value(value).map_err(|_| RootArrayUnionError::Input), &mut Control::default()).unwrap();
    assert_eq!(output.json(), r#"[{"a/~":"A"}]"#);
    let origins = &output.records()[0].origins;
    assert_eq!(origins.len(), 2);
    assert_eq!(origins[0].source_field_indices, [0]); assert_eq!(origins[1].source_field_indices, [0]);
    assert_eq!(first[0].spans.len(), 2); assert_eq!(second[0].spans[0].byte_start, 4);
    // Structural-only values receive no invented evidence.
    assert!(run(&numbers(64), &["[1]"], limits(), &mut Control::default()).unwrap().records()[0]
        .origins[0].source_field_indices.is_empty());
}
#[test]
fn final_original_max_items_refuses_a_union_that_exceeds_successful_maps() {
    assert!(run(&numbers(1), &["[1]", "[2]"], limits(), &mut Control::default()).is_err());
    assert_eq!(run(&numbers(1), &["[1]", "[1]"], limits(), &mut Control::default()).unwrap().json(), "[1]");
}
#[test]
fn final_declaration_rechecks_runtime_unicode_length_rules() {
    let schema = JsonValidationSchema::compile(r#"{"type":"array","maxItems":4,"items":{"type":"string","maxLength":2}}"#,
        CompileLimits::default()).unwrap();
    let input = [ArrayUnionChunk { chunk_id: 0, source_span: span(0, 1), json: r#"["abc"]"#, source_fields: &[] }];
    assert!(reduce(schema.root(), Sha256Digest::of_bytes(b"fixture"), limits(), input.into_iter(),
        |value| schema.validate_value(value).map_err(|_| RootArrayUnionError::Input), &mut Control::default()).is_err());
    let input = [ArrayUnionChunk { chunk_id: 0, source_span: span(0, 1), json: r#"["éé"]"#, source_fields: &[] }];
    assert_eq!(reduce(schema.root(), Sha256Digest::of_bytes(b"fixture"), limits(), input.into_iter(),
        |value| schema.validate_value(value).map_err(|_| RootArrayUnionError::Input), &mut Control::default()).unwrap().json(), r#"["éé"]"#);
}
#[test]
fn all_six_limits_refuse_without_truncation() {
    for axis in 0..6 {
        let mut cap = limits();
        match axis { 0 => cap.max_records = 1, 1 => { cap.max_records = 1; cap.max_origins = 1; },
            2 => cap.max_input_bytes = 5, 3 => cap.max_json_bytes = 4,
            4 => cap.max_nodes = 6, _ => { cap.max_json_bytes = 5; cap.max_result_bytes = 5; } }
        let second = if axis == 1 { "[1]" } else { "[2]" };
        assert!(run(&numbers(64), &["[1]", second], cap, &mut Control::default()).is_err(), "{axis}");
    }
    let mut exact = limits(); exact.max_nodes = 7; // two (array + number) maps, final array + two numbers
    assert_eq!(run(&numbers(64), &["[1]", "[2]"], exact, &mut Control::default()).unwrap().visited_nodes, 7);
}
#[test]
fn empty_arrays_succeed_without_fabricated_records_or_origins() {
    let output = run(&numbers(0), &["[]", "[]"], limits(), &mut Control::default()).unwrap();
    assert_eq!(output.json(), "[]"); assert!(output.records().is_empty()); assert_eq!(output.input_records(), 0);
}
#[test]
fn cancellation_including_the_final_publication_checkpoint_returns_no_result() {
    let mut count = Control::default();
    run(&numbers(64), &["[1,2]", "[2,3]"], limits(), &mut count).unwrap();
    for stop in [1, count.polls / 2, count.polls] {
        let error = run(&numbers(64), &["[1,2]", "[2,3]"], limits(), &mut Control { polls: 0, stop: Some(stop) }).err().unwrap();
        assert_eq!(error, RootArrayUnionError::Cancelled(DecodeCancellationKind::Deadline));
    }
}
#[test]
fn schema_root_depth_and_property_key_limits_are_explicit() {
    assert!(limits().validate_schema(&SchemaNode::Null { allowed: None }).is_err());
    let key = "k".repeat(MAX_KEY_BYTES + 1);
    let object = SchemaNode::Object { properties: BTreeMap::from([(key, SchemaNode::Null { allowed: None })]), required: Default::default() };
    assert!(limits().validate_schema(&SchemaNode::Array { items: Box::new(object), max_items: 1 }).is_err());
    let mut schema = SchemaNode::String { max_bytes: 64, allowed: None, source: SourceAnnotation::None };
    for _ in 0..=MAX_DEPTH { schema = SchemaNode::Array { items: Box::new(schema), max_items: 1 }; }
    assert!(limits().validate_schema(&schema).is_err());
}
#[test]
fn malformed_duplicate_keys_oversized_keys_and_foreign_evidence_refuse() {
    let oversized = format!("[{{\"{}\":0}}]", "x".repeat(MAX_KEY_JSON_BYTES + 1));
    assert_eq!(parse(&oversized, 1, &mut Control::default()).err(), Some(RootArrayUnionError::Limit));
    assert!(parse(r#"[{"x":1,"x":2}]"#, 100, &mut Control::default()).is_err());
    let fields = [SourceFieldEvidence { json_pointer: "/1/wrong".to_owned(), occurrence: SourceOccurrence::Anchored, spans: vec![span(0, 1)] }];
    let chunks = [ArrayUnionChunk { chunk_id: 0, source_span: span(0, 1), json: "[1]", source_fields: &fields }];
    assert!(reduce(&numbers(64), Sha256Digest::of_bytes(b"fixture"), limits(), chunks.into_iter(), |_| Ok(()), &mut Control::default()).is_err());
}
#[test]
fn quotation_is_checked_monotone_and_policy_schema_identity_is_exact() {
    let small = limits(); let mut large = small; large.max_nodes += 1;
    assert!(large.requirements().unwrap().workspace_bytes > small.requirements().unwrap().workspace_bytes);
    let a = Sha256Digest::of_bytes(b"schema A"); let b = Sha256Digest::of_bytes(b"schema B");
    assert_ne!(small.binding(a).unwrap(), small.binding(b).unwrap());
    assert_ne!(small.binding(a).unwrap(), large.binding(a).unwrap());
    let mut invalid = small; invalid.max_nodes = usize::MAX; assert!(invalid.requirements().is_err());
    let standard = RootArrayUnionLimits { max_records: 4096, max_origins: 16384, max_nodes: 262144,
        max_input_bytes: 64 << 20, max_json_bytes: 16 << 20, max_result_bytes: 16 << 20 };
    assert!(standard.requirements().unwrap().workspace_bytes < 4 * 1024 * 1024 * 1024);
}
