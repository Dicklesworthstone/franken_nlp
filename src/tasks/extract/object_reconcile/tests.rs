//! Definition-level exact JSON and original-source checks; no model inference.
use super::*;

const NUMBERS: &str = r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a"]}"#;

fn limits() -> RootObjectLimits {
    RootObjectLimits { max_properties: 16, max_origins: 64, max_input_bytes: 65536,
        max_json_bytes: 65536, max_nodes: 4096, max_result_bytes: 262144 }
}
struct Control { polls: usize, stop: usize, cause: DecodeCancellationKind }
impl Default for Control {
    fn default() -> Self { Self { polls: 0, stop: usize::MAX, cause: DecodeCancellationKind::Deadline } }
}
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.polls == self.stop).then_some(self.cause)
    }
}
fn schema(text: &str) -> RootObjectSchema {
    RootObjectSchema::derive(text, CompileLimits::default(), limits(), &mut Control::default()).unwrap()
}
fn value(text: &str) -> JsonValue {
    array_union::parse(text, limits().max_nodes, &mut Control::default()).unwrap()
}
fn span(byte_start: usize, byte_end: usize, scalar_start: usize, scalar_end: usize) -> VerifiedSourceSpan {
    VerifiedSourceSpan { byte_start, byte_end, scalar_start, scalar_end }
}
fn run(schema: &RootObjectSchema, inputs: &[&str], limits: RootObjectLimits, control: &mut Control)
    -> Result<RootObjectResult, RootObjectError> {
    let source = "x".repeat(inputs.len());
    reconcile(schema, &source, schema.binding(limits).unwrap(), limits, GroundingBudget::default(), false,
        inputs.iter().enumerate().map(|(id, json)| ObjectChunk { chunk_id: id,
            source_span: span(id, id + 1, id, id + 1), json, source_fields: &[] }), control)
}

#[test]
fn derivation_relaxes_only_root_omission_and_preserves_nested_required_and_exact_constants() {
    let original = r#"{"type":"object","additionalProperties":false,"required":["nested","rows","precise"],"properties":{"nested":{"type":"object","additionalProperties":false,"required":["name"],"properties":{"name":{"type":"string","maxLength":2}}},"rows":{"type":"array","maxItems":2,"items":{"type":"object","additionalProperties":false,"required":["value"],"properties":{"value":{"type":"integer"}}}},"precise":{"type":"number","const":12345678901234567890123456789012345678}}}"#;
    let derived = schema(original);
    let SchemaNode::Object { properties, required } = derived.original.root() else { unreachable!() };
    let SchemaNode::Object { properties: partial, required: omitted } = derived.partial.root() else { unreachable!() };
    assert_eq!(properties, partial);
    assert_eq!(required.len(), 3); assert!(omitted.is_empty());
    assert!(derived.partial.validate_value(&value("{}")).is_ok());
    assert!(derived.original.validate_value(&value("{}")).is_err());
    assert!(derived.partial_schema().contains("1.2345678901234567890123456789012345678e37"));
    for json in [r#"{"nested":{}}"#, r#"{"rows":[{}]}"#, r#"{"nested":{"name":"abc"}}"#,
        r#"{"precise":12345678901234567890123456789012345679}"#, r#"{"unknown":1}"#] {
        assert!(derived.partial.validate_value(&value(json)).is_err(), "{json}");
    }
    let output = run(&derived, &[r#"{"nested":{"name":"éé"}}"#, r#"{"rows":[{"value":7}]}"#,
        r#"{"precise":12345678901234567890123456789012345678}"#], limits(), &mut Control::default()).unwrap();
    derived.original.validate_value(&value(output.json())).unwrap();
    assert_eq!(output.properties().len(), 3);
}

#[test]
fn exact_decimals_deduplicate_without_float_rounding_and_keep_every_origin() {
    let derived = schema(r#"{"type":"object","additionalProperties":false,"properties":{"big":{"type":"number"},"precise":{"type":"number"},"unit":{"type":"number"},"zero":{"type":"number"}},"required":["big","precise","unit","zero"]}"#);
    let output = run(&derived, &[
        r#"{"big":9007199254740993,"precise":12345678901234567890123456789012345678,"unit":1.0,"zero":-0.0}"#,
        r#"{"zero":0,"unit":1e0,"precise":1.2345678901234567890123456789012345678e37,"big":9007199254740993}"#,
    ], limits(), &mut Control::default()).unwrap();
    assert_eq!(output.json(), r#"{"big":9007199254740993,"precise":1.2345678901234567890123456789012345678e37,"unit":1,"zero":0}"#);
    assert_eq!(output.input_properties, 8); assert_eq!(output.duplicate_properties, 4);
    assert_eq!(output.properties().iter().map(|property| property.json_pointer.as_str()).collect::<Vec<_>>(),
        ["/big", "/precise", "/unit", "/zero"]);
    for property in output.properties() {
        assert_eq!(property.origins.iter().map(|origin| origin.chunk_id).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(property.origins[0].source_span, span(0, 1, 0, 1));
        assert_eq!(property.origins[1].source_span, span(1, 2, 1, 2));
        assert!(property.origins.iter().all(|origin| origin.source_field_indices.is_empty()));
    }
}

#[test]
fn conflicting_atomic_properties_refuse_without_precedence_or_recursive_merging() {
    let derived = schema(r#"{"type":"object","additionalProperties":false,"properties":{"n":{"type":"number"},"object":{"type":"object","additionalProperties":false,"required":["a","b"],"properties":{"a":{"type":"integer"},"b":{"type":"integer"}}},"array":{"type":"array","items":{"type":"integer"},"maxItems":2}},"required":[]}"#);
    for pair in [
        [r#"{"n":9007199254740992}"#, r#"{"n":9007199254740993}"#],
        [r#"{"n":12345678901234567890123456789012345678}"#, r#"{"n":12345678901234567890123456789012345679}"#],
        [r#"{"object":{"a":1,"b":2}}"#, r#"{"object":{"a":1,"b":3}}"#],
        [r#"{"array":[1,2]}"#, r#"{"array":[2,1]}"#],
    ] {
        assert_eq!(run(&derived, &pair, limits(), &mut Control::default()).err(), Some(RootObjectError::Conflict));
    }
    assert_eq!(run(&derived, &[r#"{"object":{"a":1}}"#, r#"{"object":{"b":2}}"#],
        limits(), &mut Control::default()).err(), Some(RootObjectError::Input));
    let identical = run(&derived, &[r#"{"object":{"a":1,"b":2}}"#, r#"{"object":{"b":2,"a":1}}"#],
        limits(), &mut Control::default()).unwrap();
    assert_eq!(identical.json(), r#"{"object":{"a":1,"b":2}}"#);
    assert_eq!(identical.properties()[0].origins.len(), 2);
}

#[test]
fn omission_is_distinct_from_explicit_null_and_missing_required_never_becomes_success() {
    let derived = schema(r#"{"type":"object","additionalProperties":false,"properties":{"explicit":{"type":"null"},"optional":{"type":"number"}},"required":["explicit"]}"#);
    assert_eq!(run(&derived, &["{}", "{}"], limits(), &mut Control::default()).err(), Some(RootObjectError::MissingRequired));
    let output = run(&derived, &["{}", r#"{"explicit":null}"#, "{}"], limits(), &mut Control::default()).unwrap();
    assert_eq!(output.json(), r#"{"explicit":null}"#);
    assert_eq!(output.properties().len(), 1);
    assert_eq!(output.properties()[0].origins[0].chunk_id, 1);
    let optional = schema(r#"{"type":"object","additionalProperties":false,"properties":{"optional":{"type":"number"}},"required":[]}"#);
    let empty = run(&optional, &["{}", "{}"], limits(), &mut Control::default()).unwrap();
    assert_eq!(empty.json(), "{}"); assert!(empty.properties().is_empty());
    assert_eq!(empty.input_properties, 0); assert_eq!(empty.duplicate_properties, 0);
}

#[test]
fn global_source_reverification_preserves_escaped_paths_all_occurrences_and_remaining_ledgers() {
    let derived = schema(r#"{"type":"object","additionalProperties":false,"required":["a/~","nested"],"properties":{"a/~":{"type":"string","maxLength":2,"x-fnlp-source":"verbatim"},"nested":{"type":"object","additionalProperties":false,"required":["quotes"],"properties":{"quotes":{"type":"array","maxItems":1,"items":{"type":"string","maxLength":2,"x-fnlp-source":"verbatim"}}}}}}"#);
    let json = r#"{"a/~":"上海","nested":{"quotes":["上海"]}}"#;
    let evidence = |position| vec![
        SourceFieldEvidence { json_pointer: "/a~1~0".to_owned(), occurrence: SourceOccurrence::Anchored, spans: vec![position] },
        SourceFieldEvidence { json_pointer: "/nested/quotes/0".to_owned(), occurrence: SourceOccurrence::Anchored, spans: vec![position] },
    ];
    let first = evidence(span(2, 8, 1, 3)); let second = evidence(span(9, 15, 4, 6));
    let chunks = || [ObjectChunk { chunk_id: 0, source_span: span(0, 8, 0, 3), json, source_fields: &first },
        ObjectChunk { chunk_id: 1, source_span: span(8, 15, 3, 6), json, source_fields: &second }];
    let exact = GroundingBudget { max_fields: 2, max_matches: 4, max_scan_steps: 352 };
    let output = reconcile(&derived, "é上海 上海", derived.binding(limits()).unwrap(), limits(), exact,
        true, chunks().into_iter(), &mut Control::default()).unwrap();
    assert_eq!(output.verification_used(), RootObjectVerification { fields: 2, matches: 4, scan_steps: 352 });
    assert_eq!(output.original_source_fields().len(), 2);
    for field in output.original_source_fields() {
        assert_eq!(field.occurrence, SourceOccurrence::Ambiguous);
        assert_eq!(field.spans, [span(2, 8, 1, 3), span(9, 15, 4, 6)]);
    }
    for (index, property) in output.properties().iter().enumerate() {
        assert_eq!(property.origins.len(), 2);
        assert!(property.origins.iter().all(|origin| origin.source_field_indices == [index]));
    }
    for axis in 0..3 {
        let mut insufficient = exact;
        match axis { 0 => insufficient.max_fields -= 1, 1 => insufficient.max_matches -= 1,
            _ => insufficient.max_scan_steps -= 1 }
        assert!(reconcile(&derived, "é上海 上海", derived.binding(limits()).unwrap(), limits(), insufficient,
            true, chunks().into_iter(), &mut Control::default()).is_err(), "axis {axis}");
    }
    let structural = reconcile(&derived, "é上海 上海", derived.binding(limits()).unwrap(), limits(),
        GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 }, false,
        chunks().into_iter(), &mut Control::default()).unwrap();
    assert!(structural.original_source_fields().is_empty());
    assert_eq!(structural.verification_used(), RootObjectVerification { fields: 0, matches: 0, scan_steps: 0 });
    let absent = r#"{"a/~":"東京","nested":{"quotes":["東京"]}}"#;
    assert_eq!(reconcile(&derived, "é上海 上海", derived.binding(limits()).unwrap(), limits(), exact, true,
        [ObjectChunk { chunk_id: 0, source_span: span(0, 15, 0, 6), json: absent, source_fields: &[] }].into_iter(),
        &mut Control::default()).err(), Some(RootObjectError::Source));
}

#[test]
fn complete_utf8_source_coverage_rejects_gaps_overlap_missing_chunks_and_compensating_scalar_offsets() {
    let derived = schema(r#"{"type":"object","additionalProperties":false,"properties":{},"required":[]}"#);
    let valid = [span(0, 3, 0, 2), span(3, 4, 2, 3)];
    let check = |spans: &[VerifiedSourceSpan], ids: &[usize]| reconcile(&derived, "AéZ", derived.binding(limits()).unwrap(),
        limits(), GroundingBudget::default(), false, spans.iter().zip(ids).map(|(&source_span, &chunk_id)|
            ObjectChunk { chunk_id, source_span, json: "{}", source_fields: &[] }), &mut Control::default());
    assert!(check(&valid, &[0, 1]).is_ok());
    for invalid in [
        [span(1, 3, 0, 1), valid[1]],
        [valid[0], span(2, 4, 2, 3)],
        [span(0, 1, 0, 1), valid[1]],
        [span(0, 2, 0, 1), span(2, 4, 1, 3)],
        [span(0, 3, 0, 1), span(3, 4, 1, 3)],
        [valid[0], span(3, 4, 2, 4)],
    ] { assert!(check(&invalid, &[0, 1]).is_err()); }
    assert!(check(&valid[..1], &[0]).is_err());
    assert!(check(&valid, &[0, 2]).is_err());
    assert!(check(&[], &[]).is_err());
}

#[test]
fn chunk_commitments_bind_raw_json_coordinates_evidence_and_order() {
    let derived = schema(NUMBERS); let inputs = [r#"{"a":1.0}"#, r#"{"a":1}"#];
    let output = run(&derived, &inputs, limits(), &mut Control::default()).unwrap();
    let chunks = || inputs.iter().enumerate().map(|(id, json)| ObjectChunk { chunk_id: id,
        source_span: span(id, id + 1, id, id + 1), json, source_fields: &[] });
    assert!(output.matches_chunks(chunks()));
    let foreign = [SourceFieldEvidence { json_pointer: "/a".to_owned(), occurrence: SourceOccurrence::Anchored,
        spans: vec![span(0, 1, 0, 1)] }];
    for axis in 0..5 {
        let mut changed: Vec<_> = chunks().collect();
        match axis { 0 => changed[0].json = r#"{"a":1}"#,
            1 => changed[0].source_span.scalar_end += 1,
            2 => changed[0].source_fields = &foreign,
            3 => changed[0].chunk_id += 1,
            _ => changed.reverse() }
        assert!(!output.matches_chunks(changed.into_iter()), "axis {axis}");
    }
    assert!(!output.matches_chunks(chunks().take(1)));
    let public = serde_json::to_value(&output).unwrap();
    assert!(public.get("binding").is_none()); assert!(public.get("mapped_digest").is_none());
}

#[test]
fn every_reconciliation_limit_is_nonrenewable_and_exact_node_and_byte_edges_are_accepted() {
    let derived = schema(NUMBERS); let inputs = [r#"{"a":1}"#, r#"{"b":2}"#];
    let baseline = run(&derived, &inputs, limits(), &mut Control::default()).unwrap();
    assert_eq!(baseline.visited_nodes, 7);
    assert_eq!(baseline.input_json_bytes, 14);
    let mut exact = limits(); exact.max_nodes = 7; exact.max_input_bytes = 14; exact.max_json_bytes = 13;
    assert_eq!(run(&derived, &inputs, exact, &mut Control::default()).unwrap().json(), r#"{"a":1,"b":2}"#);
    for axis in 0..6 {
        let mut short = exact;
        let mut pair = inputs;
        match axis {
            0 => short.max_properties = 1,
            1 => { short.max_properties = 1; short.max_origins = 1; pair[1] = pair[0]; },
            2 => short.max_input_bytes -= 1,
            3 => short.max_json_bytes -= 1,
            4 => short.max_nodes -= 1,
            _ => short.max_result_bytes = short.max_json_bytes,
        }
        assert!(run(&derived, &pair, short, &mut Control::default()).is_err(), "axis {axis}");
    }
}

#[test]
fn invalid_schemas_values_limits_and_foreign_evidence_refuse_before_any_success() {
    for source in [r#"{"type":"array","items":{"type":"number"},"maxItems":1}"#,
        r#"{"type":"object","properties":{"a":{"type":"number"}},"required":["missing"],"additionalProperties":false}"#,
        r#"{"type":"object","properties":{},"additionalProperties":true}"#] {
        assert!(RootObjectSchema::derive(source, CompileLimits::default(), limits(), &mut Control::default()).is_err());
    }
    let derived = schema(NUMBERS);
    for json in [r#"{"a":1,"a":2}"#, r#"{"extra":1}"#, r#"{"a":"wrong"}"#, "[]", "{"] {
        assert!(run(&derived, &[json], limits(), &mut Control::default()).is_err());
    }
    let foreign = [SourceFieldEvidence { json_pointer: "/foreign".to_owned(), occurrence: SourceOccurrence::Anchored,
        spans: vec![span(0, 1, 0, 1)] }];
    assert!(reconcile(&derived, "x", derived.binding(limits()).unwrap(), limits(), GroundingBudget::default(), false,
        [ObjectChunk { chunk_id: 0, source_span: span(0, 1, 0, 1), json: r#"{"a":1}"#, source_fields: &foreign }].into_iter(),
        &mut Control::default()).is_err());
    for axis in 0..7 {
        let mut invalid = limits();
        match axis { 0 => invalid.max_properties = 0, 1 => invalid.max_origins = 0,
            2 => invalid.max_input_bytes = usize::MAX, 3 => invalid.max_json_bytes = 0,
            4 => invalid.max_nodes = usize::MAX, 5 => invalid.max_result_bytes = 0,
            _ => invalid.max_origins = invalid.max_properties - 1 }
        assert_eq!(invalid.workspace_bytes().err(), Some(RootObjectError::InvalidLimits));
    }
}

#[test]
fn original_schema_and_every_policy_limit_bind_even_when_partial_schema_bytes_are_identical() {
    let derived = schema(NUMBERS);
    let spaced = schema(&format!(" {NUMBERS}\n"));
    assert_eq!(derived.partial_schema(), spaced.partial_schema());
    assert_ne!(derived.binding(limits()).unwrap(), spaced.binding(limits()).unwrap());
    let no_required = schema(&NUMBERS.replace(r#""required":["a"]"#, r#""required":[]"#));
    assert_eq!(derived.partial_schema(), no_required.partial_schema());
    assert_ne!(derived.binding(limits()).unwrap(), no_required.binding(limits()).unwrap());
    for axis in 0..6 {
        let mut changed = limits();
        match axis { 0 => changed.max_properties += 1, 1 => changed.max_origins += 1,
            2 => changed.max_input_bytes += 1, 3 => changed.max_json_bytes += 1,
            4 => changed.max_nodes += 1, _ => changed.max_result_bytes += 1 }
        assert_ne!(derived.binding(limits()).unwrap(), derived.binding(changed).unwrap());
        assert!(changed.workspace_bytes().unwrap() >= limits().workspace_bytes().unwrap());
    }
}

#[test]
fn derivation_and_final_reconciliation_preserve_every_original_cancellation_cause() {
    let mut derivation = Control::default();
    RootObjectSchema::derive(NUMBERS, CompileLimits::default(), limits(), &mut derivation).unwrap();
    let derived = schema(NUMBERS); let mut reconciliation = Control::default();
    run(&derived, &[r#"{"a":1}"#, r#"{"a":1,"b":2}"#], limits(), &mut reconciliation).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop in [1, derivation.polls / 2, derivation.polls] {
            let mut control = Control { polls: 0, stop, cause };
            assert_eq!(RootObjectSchema::derive(NUMBERS, CompileLimits::default(), limits(), &mut control).err(),
                Some(RootObjectError::Cancelled(cause)));
            assert_eq!(control.polls, stop);
        }
        for stop in [1, reconciliation.polls / 2, reconciliation.polls] {
            let mut control = Control { polls: 0, stop, cause };
            assert_eq!(run(&derived, &[r#"{"a":1}"#, r#"{"a":1,"b":2}"#], limits(), &mut control).err(),
                Some(RootObjectError::Cancelled(cause)));
            assert_eq!(control.polls, stop);
        }
    }
}
