//! Pinned compilation and independent finalizers; native records are synthetic.
use super::*;
use crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE;

fn limits(rows: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step: rows } }
fn selected(p: &SourceTaskPlanner, r: &SourceTaskRequest) -> PreparedInt4SourceTask {
    prepare(p, r).with_selected_rows(limits(7)).unwrap()
}
fn raw(p: &PreparedInt4SourceTask, r: &SourceTaskRequest, source: &str, json: &str) -> Int4ExtractRun {
    let mut raw = synthetic(p, r, source, json);
    let work = Int4Work::for_sequence(p.group_width(), 0, p.prompt_tokens() + 1, 3).unwrap();
    raw.execution = p.extraction.execution_version().to_owned();
    raw.model_work = work;
    raw.result.output.forward_positions = work.forward_positions;
    raw.result.output.projected_logits = work.projected_logits;
    raw
}

#[test]
fn each_task_binds_the_selected_cap_and_preserves_all_other_authority() {
    let p = planner();
    for r in [ner("é上海"), keyphrases("é上海"), summary("é上海"), answer(&[("p", "é上海")])] {
        let dense = prepare(&p, &r); let sparse = selected(&p, &r);
        assert_eq!(dense.execution_version(), INT4_SOURCE_EXECUTION);
        assert!(dense.selected_rows().is_none());
        assert_eq!(sparse.execution_version(), INT4_SPARSE_SOURCE_EXECUTION);
        assert_eq!(sparse.selected_rows(), Some(limits(7)));
        assert_eq!(dense.task_budget(), sparse.task_budget());
        assert_eq!(dense.prompt_tokens(), sparse.prompt_tokens());
        assert_eq!(dense.request_digest(), sparse.request_digest());
        assert_eq!(dense.max_result_bytes(), sparse.max_result_bytes());
        assert_eq!(dense.max_kv_bytes(), sparse.max_kv_bytes());
        assert_eq!(dense.execution_identity().prompt_digest, sparse.execution_identity().prompt_digest);
        assert_eq!(dense.execution_identity().schema_digest, sparse.execution_identity().schema_digest);
        assert_eq!(dense.execution_identity().task_spec, sparse.execution_identity().task_spec);
        assert_eq!(dense.planned_work().forward_positions, sparse.planned_work().forward_positions);
        assert_eq!(dense.planned_work().attention_pairs, sparse.planned_work().attention_pairs);
        assert_eq!(sparse.planned_work().projected_logits, 7 * u64::from(r.budget().max_output_tokens));
        assert!(sparse.planned_work().projections.dot_products < dense.planned_work().projections.dot_products);
        assert_ne!(dense.comparison_policy_digest(), sparse.comparison_policy_digest());
        assert!(sparse.verify_identity(dense.execution_identity()).is_err());
        sparse.verify_identity(sparse.execution_identity()).unwrap();
        dense.require_full_projection().unwrap();
        assert!(sparse.require_full_projection().is_err());
    }
}

#[test]
fn invalid_repeated_and_full_vocabulary_choices_cannot_disguise_the_strategy() {
    let p = planner(); let r = ner("Alice");
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert!(prepare(&p, &r).with_selected_rows(limits(cap)).is_err());
    }
    assert!(selected(&p, &r).with_selected_rows(limits(7)).is_err());
    let dense = prepare(&p, &r);
    let full = prepare(&p, &r).with_selected_rows(limits(NANBEIGE_VOCAB_SIZE)).unwrap();
    assert_eq!(dense.planned_work(), full.planned_work());
    assert_ne!(dense.execution_identity(), full.execution_identity());
    assert_ne!(dense.comparison_policy_digest(), full.comparison_policy_digest());
    assert!(full.require_full_projection().is_err());
    assert_eq!(full.execution_version(), INT4_SPARSE_SOURCE_EXECUTION);
}

#[test]
fn selected_source_finalizer_and_cap_are_sealed_independently_of_group_comparison() {
    let p = planner(); let r = ner("Alice"); let sparse = selected(&p, &r);
    let extraction = prepare(&p, &r).extraction.with_selected_rows(limits(7)).unwrap();
    assert!(sparse.verify_identity(extraction.execution_identity()).is_err());
    let finalized = extraction.with_finalizer_version(INT4_SPARSE_SOURCE_EXECUTION).unwrap();
    sparse.verify_identity(finalized.execution_identity()).unwrap();
    let id = identity_with_width(&p, &r, 127);
    let other_width = p.plan_int4_with_control(&r, &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), 127, &mut Continue).unwrap().with_selected_rows(limits(7)).unwrap();
    assert_eq!(sparse.comparison_policy_digest(), other_width.comparison_policy_digest());
    assert!(sparse.verify_identity(other_width.execution_identity()).is_err());
    let other_cap = prepare(&p, &r).with_selected_rows(limits(8)).unwrap();
    assert_ne!(sparse.comparison_policy_digest(), other_cap.comparison_policy_digest());
    assert_ne!(sparse.comparison_policy_digest(), selected(&p, &keyphrases("Alice")).comparison_policy_digest());
}

#[test]
fn selected_ner_and_keyphrases_preserve_all_unicode_occurrences_and_model_order() {
    let p = planner(); let source = "é上海 Rust 上海";
    let r = ner(source); let plan = selected(&p, &r);
    let run = plan.finish(raw(&plan, &r, source, r#"[{"text":"上海","type":"location"}]"#)).unwrap();
    assert_eq!(run.execution, INT4_SPARSE_SOURCE_EXECUTION);
    let SourceTaskResult::Ner(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.entities[0].occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(result.entities[0].spans.len(), 2);
    for s in &result.entities[0].spans {
        validate_source_span(source, "上海", SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
    }
    let r = keyphrases(source); let plan = selected(&p, &r);
    let run = plan.finish(raw(&plan, &r, source, r#"["Rust","上海","上海"]"#)).unwrap();
    let SourceTaskResult::Keyphrases(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.phrases.iter().map(|p| (p.rank, p.text.as_str())).collect::<Vec<_>>(), [(1, "Rust"), (2, "上海")]);
    assert_eq!(result.phrases[1].occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(result.phrases[1].spans.len(), 2);
}

#[test]
fn selected_summaries_and_answers_keep_citation_abstention_and_passage_contracts() {
    let p = planner(); let r = summary("Alice"); let plan = selected(&p, &r);
    let run = plan.finish(raw(&plan, &r, "Alice", r#"[{"citations":["Alice"],"text":"An unverified claim."}]"#)).unwrap();
    let SourceTaskResult::Summarize(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.citation_guarantee, CitationGuarantee::StructuralSourceMembership);
    assert_eq!(result.semantic_support, SummarySemanticSupport::NotAssessed);
    assert!(plan.finish(raw(&plan, &r, "Alice", r#"[{"citations":[],"text":"A claim."}]"#)).is_err());
    let r = answer(&[("one", "é上海"), ("two", "上海")]); let plan = selected(&p, &r);
    let run = plan.finish(raw(&plan, &r, "é上海\n\n上海",
        r#"{"answer":"A location.","answerable":true,"citations":["上海"]}"#)).unwrap();
    let SourceTaskResult::Answer(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.calibration, AnswerCalibration::Uncalibrated);
    assert_eq!(result.citations[0].occurrence, SourceOccurrence::Ambiguous);
    let spans = &result.citations[0].spans;
    assert_eq!(spans.len(), 2);
    assert_eq!((spans[0].passage_id.as_str(), spans[0].span.byte_start, spans[0].span.scalar_start), ("one", 2, 1));
    assert_eq!((spans[1].passage_id.as_str(), spans[1].span.byte_start), ("two", 0));
    let r = answer(&[("a", "ab"), ("b", "cd")]); let plan = selected(&p, &r);
    assert!(plan.finish(raw(&plan, &r, "ab\n\ncd",
        r#"{"answer":"A claim.","answerable":true,"citations":["b\n\nc"]}"#)).is_err());
    let run = plan.finish(raw(&plan, &r, "ab\n\ncd", r#"{"answer":"","answerable":false,"citations":[]}"#)).unwrap();
    let SourceTaskResult::Answer(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.status, AnswerStatus::Abstained); assert!(result.answer.is_none());
    for json in [r#"{"answer":"ab","answerable":true,"citations":[]}"#,
        r#"{"answer":"ab","answerable":false,"citations":[]}"#] {
        assert!(plan.finish(raw(&plan, &r, "ab\n\ncd", json)).is_err());
    }
}

#[test]
fn selected_tasks_reject_dense_envelopes_and_corrupt_independent_evidence() {
    let p = planner();
    for (r, json) in [(ner("Alice"), r#"[{"text":"Alice","type":"person"}]"#),
        (keyphrases("Alice"), r#"["Alice"]"#),
        (summary("Alice"), r#"[{"citations":["Alice"],"text":"A name."}]"#),
        (answer(&[("p", "Alice")]), r#"{"answer":"Alice","answerable":true,"citations":["Alice"]}"#)] {
        let plan = selected(&p, &r); let dense = prepare(&p, &r);
        assert!(plan.finish(synthetic(&dense, &r, "Alice", json)).is_err());
        assert!(dense.finish(raw(&plan, &r, "Alice", json)).is_err());
        for axis in 0..6 {
            let mut bad = raw(&plan, &r, "Alice", json);
            match axis { 0 => bad.result.source_fields.clear(),
                1 => bad.result.source_fields[0].json_pointer = "/other".to_owned(),
                2 => bad.result.source_fields[0].spans[0].scalar_end += 1,
                3 => bad.result.output.numerics_profile = "hf-bf16-eager".to_owned(),
                4 => bad.group_width = 127, _ => bad.result.task_spec_version = "different-v1".to_owned() }
            assert!(plan.finish(bad).is_err());
        }
    }
}

#[test]
fn complete_selected_envelope_still_obeys_exact_result_limits_and_privacy() {
    let p = planner(); let mut r = ner("Alice"); let plan = selected(&p, &r);
    let json = r#"[{"text":"Alice","type":"person"}]"#;
    let run = plan.finish(raw(&plan, &r, "Alice", json)).unwrap();
    let exact = canonjson::canonical_bytes(&run).unwrap().len() as u64;
    for (cap, succeeds) in [(exact, true), (exact - 1, false)] {
        if let SourceTaskRequest::Ner { budget, .. } = &mut r { budget.max_output_bytes = cap; }
        let plan = selected(&p, &r);
        assert_eq!(plan.finish(raw(&plan, &r, "Alice", json)).is_ok(), succeeds);
    }
    let encoded = canonjson::canonical_string(&run).unwrap();
    for absent in ["prompt_digest", "taskir_digest", "request_digest", "confidence", "logical_model_digest"] {
        assert!(!encoded.contains(absent));
    }
}
