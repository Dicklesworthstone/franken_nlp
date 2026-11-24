//! Source-defined admission/accounting regressions; no native-success or
//! quality claim. Executable validation remains a separate controller gate.
use super::*;
use crate::{
    native_engine::{constrained::JsonDecodeOutput, decode::DecodeCancellationKind,
        lmhead::{NANBEIGE_VOCAB_SIZE, scoring::ScoringWork}},
    tasks::{extract::{ExtractResult, ExtractionGrounding, semantic::{
        SemanticFieldResult, SemanticFieldStatus, SemanticNotChecked, SemanticVerificationReceipt,
        CLAIM_RENDER_VERSION, SEMANTIC_VERIFICATION_VERSION}},
        ir::ScoreSpace, judge::FAITHFULNESS_VERSION},
};

fn fixture() -> (Int8SemanticHostLimits, TaskBudget, TaskBudget, SemanticLimits, JudgeLimits) {
    let kv = Int8MemoryRequirement::for_context(256).unwrap().kv_bytes;
    let extraction = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: kv };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 1, max_kv_bytes: kv };
    let semantic = SemanticLimits { max_fields: 3, max_claim_bytes: 128,
        max_total_claim_bytes: 1024, max_template_bytes: 1024, max_walk_nodes: 128,
        max_input_json_bytes: 8192, max_total_prompt_tokens: 2048,
        max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 };
    let planning = JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
        max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() };
    let host = Int8SemanticHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65_536 } },
        masks: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_mask_node_visits: 32_000, preparation_reserve_bytes: 256 << 20,
    };
    (host, extraction, judge, semantic, planning)
}

#[test]
fn both_stages_admit_the_full_reused_native_context() {
    let (host, extraction, judge, semantic, planning) = fixture();
    let required = host.requirements(extraction, judge, semantic, planning, 1024).unwrap();
    assert_eq!(required.native.kv_bytes, extraction.max_kv_bytes);
    for stage in 0..2 {
        let (mut extraction, mut judge) = (extraction, judge);
        if stage == 0 { extraction.max_kv_bytes -= 1; } else { judge.max_kv_bytes -= 1; }
        assert!(host.requirements(extraction, judge, semantic, planning, 1024).is_err());
    }
    let mut larger = host; larger.native.context_tokens += 1;
    assert!(larger.requirements(extraction, judge, semantic, planning, 1024).is_err());
}

#[test]
fn every_retained_field_plan_prices_source_candidates_results_and_json_pointers() {
    let (host, extraction, judge, mut semantic, planning) = fixture();
    semantic.max_fields = 1;
    let one = host.requirements(extraction, judge, semantic, planning, 16_384).unwrap();
    semantic.max_fields = 4;
    let four = host.requirements(extraction, judge, semantic, planning, 16_384).unwrap();
    assert_eq!(one.native, four.native);
    assert_eq!(one.native_workspace_bytes, four.native_workspace_bytes);
    assert_eq!(one.output_bytes, four.output_bytes);
    let per_field = 16_384 + 256 * planning.max_total_candidate_tokens as u64
        + 2 * 32 * 33 * planning.per_head.max_candidate_id_bytes as u64
        + 65_536 + 4 * judge.max_output_bytes.min(planning.max_output_bytes)
        + 2 * semantic.max_input_json_bytes as u64;
    assert_eq!(four.preparation_bytes - one.preparation_bytes, 3 * per_field);
    assert!(four.total_bytes().unwrap() > one.total_bytes().unwrap());
}

#[test]
fn final_output_is_independent_of_all_retained_sources_and_complete_judge_results() {
    let (host, extraction, judge, semantic, planning) = fixture();
    let small = host.requirements(extraction, judge, semantic, planning, 1024).unwrap();
    let large = host.requirements(extraction, judge, semantic, planning, 32_768).unwrap();
    assert_eq!(small.output_bytes, large.output_bytes);
    assert_eq!(large.preparation_bytes - small.preparation_bytes,
        (semantic.max_fields as u64 + 2) * (32_768 - 1024));
    let mut wider = judge; wider.max_output_bytes *= 2;
    let mut plans = planning; plans.max_output_bytes *= 2;
    let larger_results = host.requirements(extraction, wider, semantic, plans, 1024).unwrap();
    assert_eq!(small.output_bytes, larger_results.output_bytes);
    assert_eq!(larger_results.preparation_bytes - small.preparation_bytes,
        4 * semantic.max_fields as u64 * judge.max_output_bytes);
    let total = small.total_bytes().unwrap();
    for part in [small.native.kv_bytes, small.native_workspace_bytes,
        small.preparation_bytes, small.output_bytes] { assert!(total > part); }
    assert!(Int8SemanticHostRequirements { preparation_bytes: u64::MAX, ..small }.total_bytes().is_err());
}

#[test]
fn missing_admission_controls_and_arithmetic_overflow_refuse_before_execution() {
    for axis in 0..15 {
        let (mut host, extraction, judge, mut semantic, mut planning) = fixture();
        match axis {
            0 => host.preparation_reserve_bytes = 0,
            1 => host.native.allocator_reserve_bytes = 0,
            2 => host.max_mask_node_visits = 0,
            3 => host.masks.max_trie_node_visits = 0,
            4 => host.masks.checkpoint_interval_nodes = 0,
            5 => semantic.max_fields = 0,
            6 => planning.max_total_candidate_tokens = 0,
            7 => semantic.max_output_bytes = u64::MAX,
            8 => host.preparation_reserve_bytes = u64::MAX,
            9 => planning.max_output_bytes = 0,
            10 => semantic.max_claim_bytes = 0,
            11 => semantic.max_total_projected_logits = 0,
            12 => planning.max_total_projected_logits = 0,
            13 => host.native.run.cleanup_reserve_bytes = 0,
            _ => semantic.max_fields = 4097,
        }
        assert!(host.requirements(extraction, judge, semantic, planning, 1024).is_err(), "axis {axis}");
    }
    let (host, extraction, judge, semantic, planning) = fixture();
    assert!(host.requirements(extraction, judge, semantic, planning, 0).is_err());
    if usize::BITS >= 64 {
        assert!(host.requirements(extraction, judge, semantic, planning, usize::MAX).is_err());
    }
}

// Synthetic private receipt only: the public host consumes a sealed native
// plan and cannot accept this value or an injected model result.
fn completion() -> (Int8SemanticExtractionRun, Int8Work) {
    let extraction = Int8Work::for_sequence(0, 8, 3 * NANBEIGE_VOCAB_SIZE).unwrap();
    let actual = Int8Work::for_sequence(0, 7, 2 * NANBEIGE_VOCAB_SIZE).unwrap();
    let run = Int8SemanticExtractionRun {
        schema_version: 1, execution: INT8_SEMANTIC_EXECUTION, numerics_profile: STRICT_INT8_PROFILE,
        extraction_reserved_work: extraction, semantic_model_work: Int8Work::default(),
        reserved_model_work: extraction, model_work: actual, judge_head_count: 0, rewound_positions: 0,
        input_encoding: None,
        extraction: Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.into(), model_work: actual,
            result: ExtractResult { schema_version: 1, task_spec_version: "extract-v1".into(),
                score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::NotRequested,
                source_fields: Vec::new(),
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.into(),
                    token_ids: vec![1, 2], json: "1".into(), forward_positions: actual.forward_positions,
                    projected_logits: actual.projected_logits, mask_node_visit_charge: 1 } } },
        semantic: SemanticVerificationReceipt {
            schema_version: 1, algorithm: SEMANTIC_VERIFICATION_VERSION.into(),
            claim_rendering: CLAIM_RENDER_VERSION.into(), judge_algorithm: FAITHFULNESS_VERSION.into(),
            specification_revision: "host-accounting-fixture".into(),
            calibration: "experimental_uncalibrated_diagnostic_only".into(), same_model_correlated: true,
            attempted_fields: 0, checked_fields: 0, not_checked_fields: 1,
            fields: vec![SemanticFieldResult { json_pointer: "$".into(), status: SemanticFieldStatus::NotChecked,
                not_checked: Some(SemanticNotChecked::NoClaimRule), claim_rule_id: None, rendered_claim: None,
                same_model_correlated: None, judge: None }],
            work: ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 },
        },
    };
    (run, extraction)
}
fn axis(work: &mut Int8Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn completion_never_refunds_any_original_extraction_work_axis_after_early_eos() {
    let (run, extraction) = completion();
    check_completion(&run, extraction, extraction).unwrap();
    assert!(check_completion(&run, extraction, run.model_work).is_err());
    for index in 0..5 {
        let mut below = extraction; *axis(&mut below, index) -= 1;
        assert!(check_completion(&run, extraction, below).is_err(), "ceiling axis {index}");
        for receipt in 0..3 {
            let (mut corrupt, original) = completion();
            let work = match receipt { 0 => &mut corrupt.extraction_reserved_work,
                1 => &mut corrupt.reserved_model_work, _ => &mut corrupt.model_work };
            *axis(work, index) += 1;
            assert!(check_completion(&corrupt, original, original).is_err(), "receipt {receipt} axis {index}");
        }
        let (mut overflow, original) = completion();
        *axis(&mut overflow.semantic_model_work, index) = u64::MAX;
        assert!(check_completion(&overflow, original, original).is_err(), "overflow axis {index}");
    }
}

#[test]
fn completion_cannot_publish_another_backend_or_promote_correlated_diagnostics() {
    for axis in 0..6 {
        let (mut run, extraction) = completion();
        match axis {
            0 => run.schema_version += 1,
            1 => run.execution = "other-execution",
            2 => run.numerics_profile = "hf-bf16-eager",
            3 => run.extraction.execution = "selected-row-extraction".into(),
            4 => run.semantic.same_model_correlated = false,
            _ => run.semantic.calibration = "qualified".into(),
        }
        assert!(check_completion(&run, extraction, extraction).is_err(), "axis {axis}");
    }
}

#[test]
fn all_typed_cancellation_causes_remain_available_without_private_diagnostics() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let error = HostedError::SemanticExtraction(Int8SemanticError::Cancelled(cause));
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
        let HostedError::SemanticExtraction(inner) = error else { panic!("wrong cause"); };
        assert_eq!(inner.cancellation(), Some(cause));
    }
}

#[test]
fn complete_native_plan_vocabulary_and_output_cross_one_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<SemanticInput>();
    send::<PreparedInt8SemanticExtraction>();
    send::<Int8SemanticExtractionRun>();
    send::<Arc<ExtractionVocabulary>>();
    send::<HostedOutput<Int8SemanticExtractionRun>>();
    let _entry = NlpEngine::execute_int8_semantic_extract;
}

#[test]
fn compact_semantic_admission_adds_one_workspace_without_discounting_retained_fields() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let (host, extraction, judge, semantic, planning) = fixture();
    let byte = host.requirements(extraction, judge, semantic, planning, 1024).unwrap();
    assert_eq!(byte, host.requirements_with_encoding(extraction, judge, semantic, planning,
        1024, &UntrustedEncoding::ByteFallback).unwrap());
    let limits = SafeBpeLimits { max_input_bytes: 65_536, max_output_tokens: 8192,
        max_pair_checks: 1_000_000, max_pair_bytes: 8_000_000 };
    let policy = UntrustedEncoding::SafeBpeV1(limits);
    let compact = host.requirements_with_encoding(extraction, judge, semantic, planning, 1024, &policy).unwrap();
    assert_eq!(compact.native, byte.native);
    assert_eq!(compact.native_workspace_bytes, byte.native_workspace_bytes);
    assert_eq!(compact.output_bytes, byte.output_bytes);
    assert_eq!(compact.preparation_bytes - byte.preparation_bytes, limits.requirements().unwrap().workspace_bytes);
    let mut more = semantic; more.max_fields *= 2;
    let more_byte = host.requirements(extraction, judge, more, planning, 1024).unwrap();
    let more_compact = host.requirements_with_encoding(extraction, judge, more, planning, 1024, &policy).unwrap();
    assert_eq!(more_compact.preparation_bytes - more_byte.preparation_bytes,
        compact.preparation_bytes - byte.preparation_bytes);
    assert_eq!(more_compact.preparation_bytes - compact.preparation_bytes,
        more_byte.preparation_bytes - byte.preparation_bytes);
    let bad = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_pair_checks: 0, ..limits });
    assert!(host.requirements_with_encoding(extraction, judge, semantic, planning, 1024, &bad).is_err());
}
