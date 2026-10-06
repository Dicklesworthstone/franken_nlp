//! Private synthetic finalizer fixtures; no native/model execution is claimed.
use super::*;
fn limits(rows: usize) -> selected_head::Int8JsonSparseLimits {
    selected_head::Int8JsonSparseLimits { max_rows_per_step: rows }
}
fn sparse_raw(p: &Int8ExtractPlan, json: &str, rows: u64) -> Int8JsonRun {
    let mut run = raw(p, json);
    let work = p.completed_work(2, rows).unwrap();
    run.execution = selected_head::INT8_SPARSE_JSON_EXECUTION.to_owned();
    run.model_work = work; run.output.forward_positions = work.forward_positions;
    run.output.projected_logits = work.projected_logits; run
}

#[test]
fn head_choice_is_sealed_without_changing_prompt_schema_controls_or_profile() {
    let dense = plan(BOOLEAN);
    let old_identity = dense.execution_identity().clone(); let old_work = dense.planned_work();
    let prompt = dense.extraction.prompt.clone();
    let options = canonjson::canonical_bytes(dense.options()).unwrap();
    assert!(dense.selected_rows().is_none()); assert_eq!(dense.execution_version(), INT8_EXTRACT_VERSION);
    let sparse = dense.with_selected_rows(limits(7)).unwrap();
    assert_eq!(sparse.extraction.prompt, prompt);
    assert_eq!(canonjson::canonical_bytes(sparse.options()).unwrap(), options);
    assert_eq!(sparse.planned_work().forward_positions, old_work.forward_positions);
    assert_eq!(sparse.planned_work().attention_pairs, old_work.attention_pairs);
    assert_eq!(sparse.planned_work().projected_logits, 56);
    assert!(sparse.verify_identity(&old_identity).is_err());
    sparse.verify_identity(sparse.execution_identity()).unwrap();
    let mut old = serde_json::to_value(&old_identity).unwrap();
    let mut new = serde_json::to_value(sparse.execution_identity()).unwrap();
    old.as_object_mut().unwrap().remove("decision_policy_digest");
    new.as_object_mut().unwrap().remove("decision_policy_digest");
    assert_eq!(old, new);
    assert_ne!(sparse.execution_identity().decision_policy_digest,
        plan(BOOLEAN).with_selected_rows(limits(8)).unwrap().execution_identity().decision_policy_digest);
}

#[test]
fn invalid_or_repeated_head_choices_are_refused() {
    for n in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert!(plan(BOOLEAN).with_selected_rows(limits(n)).is_err());
    }
    assert!(plan(BOOLEAN).with_selected_rows(limits(3)).unwrap().with_selected_rows(limits(4)).is_err());
    assert_eq!(plan(BOOLEAN).planned_work().projected_logits, 8 * NANBEIGE_VOCAB_SIZE as u64);
}

#[test]
fn full_and_sparse_execution_envelopes_cannot_be_interchanged() {
    let dense = plan(BOOLEAN); let sparse = plan(BOOLEAN).with_selected_rows(limits(3)).unwrap();
    assert!(sparse.finalize(raw(&sparse, "true")).is_err());
    assert!(dense.finalize(sparse_raw(&sparse, "true", 3)).is_err());
    let completed = sparse.finalize(sparse_raw(&sparse, "true", 3)).unwrap();
    assert_eq!(completed.execution, INT8_SPARSE_EXTRACT_VERSION);
    assert_eq!(completed.result.output.projected_logits, 3);
    assert_eq!(completed.model_work, sparse.completed_work(2, 3).unwrap());
    sparse.verify_completed(&completed).unwrap(); assert!(dense.verify_completed(&completed).is_err());
    let mut relabeled = completed; relabeled.execution = INT8_EXTRACT_VERSION.to_owned();
    assert!(sparse.verify_completed(&relabeled).is_err()); assert!(dense.verify_completed(&relabeled).is_err());
}

#[test]
fn completed_work_requires_a_scored_eos_and_checks_every_native_axis() {
    let p = plan(BOOLEAN).with_selected_rows(limits(3)).unwrap();
    for count in 1..=8 {
        for rows in [count, count * 3] {
            let work = p.completed_work(count, rows as u64).unwrap();
            assert_eq!(work, Int8Work::for_sequence(0, p.prompt_tokens() + count - 1, rows).unwrap());
        }
        assert!(p.completed_work(count, (count - 1) as u64).is_err());
        assert!(p.completed_work(count, (count * 3 + 1) as u64).is_err());
    }
    assert!(p.completed_work(0, 0).is_err()); assert!(p.completed_work(9, 9).is_err());
    assert!(p.completed_work(2, u64::MAX).is_err());
    for axis in 0..6 {
        let mut run = sparse_raw(&p, "true", 3);
        match axis { 0 => run.model_work.forward_positions += 1, 1 => run.model_work.attention_pairs += 1,
            2 => run.model_work.projections.dot_products += 1, 3 => run.model_work.projections.multiply_accumulates += 1,
            4 => run.output.projected_logits += 1, _ => run.output.forward_positions += 1 }
        assert!(p.finalize(run).is_err());
    }
}

#[test]
fn sparse_source_evidence_keeps_all_occurrences_and_rejects_off_source_bytes() {
    let source = "é Alice 上海 Alice";
    let p = source_plan(source, &[]).with_selected_rows(limits(3)).unwrap();
    let run = p.finalize(sparse_raw(&p, r#""Alice""#, 3)).unwrap();
    assert_eq!(run.result.grounding, ExtractionGrounding::SourceMembership);
    let evidence = &run.result.source_fields[0];
    assert_eq!(evidence.occurrence, SourceOccurrence::Ambiguous); assert_eq!(evidence.spans.len(), 2);
    for span in &evidence.spans {
        assert_eq!(&source[span.byte_start..span.byte_end], "Alice");
        assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
    }
    p.verify_completed(&run).unwrap();
    assert!(p.finalize(sparse_raw(&p, r#""Mallory""#, 3)).is_err());
    let mut corrupt = run; corrupt.result.source_fields.clear();
    assert!(p.verify_completed(&corrupt).is_err());
}

#[test]
fn sparse_results_preserve_exact_decimal_strings_and_whole_envelope_limits() {
    let decimal = "1.2345678901234567890123456789012345678e37";
    let mut p = plan(r#"{"type":"number"}"#).with_selected_rows(limits(3)).unwrap();
    let run = p.finalize(sparse_raw(&p, decimal, 3)).unwrap();
    assert_eq!(run.result.output.json, decimal);
    let cap = canonjson::canonical_bytes(&run).unwrap().len() as u64;
    p.extraction.max_result_bytes = cap;
    assert!(p.finalize(sparse_raw(&p, decimal, 3)).is_ok());
    p.extraction.max_result_bytes -= 1;
    assert!(p.finalize(sparse_raw(&p, decimal, 3)).is_err());
}

#[test]
fn sparse_selection_does_not_weaken_profile_control_or_schema_checks() {
    let p = plan(BOOLEAN).with_selected_rows(limits(3)).unwrap();
    for axis in 0..5 {
        let mut run = sparse_raw(&p, "true", 3);
        match axis { 0 => run.output.numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(),
            1 => { run.output.token_ids.pop(); }, 2 => run.output.token_ids[0] = 3,
            3 => run.output.json = "null".to_owned(), _ => run.output.json = "tru".to_owned() }
        assert!(p.finalize(run).is_err());
    }
}

#[test]
fn selected_rows_do_not_expand_the_64_bit_integer_domain() {
    let p = plan(r#"{"type":"integer"}"#).with_selected_rows(limits(3)).unwrap();
    for integer in ["-9223372036854775808", "9007199254740993", "18446744073709551615"] {
        let run = p.finalize(sparse_raw(&p, integer, 3)).unwrap();
        assert_eq!(run.result.output.json, integer); p.verify_completed(&run).unwrap();
    }
    for outside in ["-9223372036854775809", "18446744073709551616", "1.2345678901234567890123456789012345678e37"] {
        assert!(p.finalize(sparse_raw(&p, outside, 3)).is_err());
    }
}
