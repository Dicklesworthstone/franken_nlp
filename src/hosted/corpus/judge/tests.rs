//! Admission/ledger/transport invariants, not synthetic native inference passes.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{portable_int8::ProjectionWork, strict_int8::STRICT_INT8_EXECUTION},
    tasks::judge::{JudgeRequest, PairwisePolicy, RubricDefinition, RubricPolicy, FaithfulnessPolicy}};

fn work() -> Int8Work { Int8Work::for_sequence(0, 16, 3 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn twice() -> Int8Work { work().checked_add(work()).unwrap() }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 64,
        max_output_bytes: 8192, max_grammar_states: 4096, max_kv_bytes: 65536 }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "nanbeige42-int8-v1".to_owned(),
        packing_set_digest: d, tokenizer_digest: d, template_digest: d, task_spec: "judge-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "judge-full-vocab-scored-eos-v1".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}

#[test]
fn planning_previews_do_not_spend_or_mutate_native_allowances() {
    let ledger = WorkLedger::new(twice());
    for _ in 0..3 { assert_eq!(ledger.preview(work()).unwrap(), work()); }
    assert_eq!(ledger.spent, Int8Work::default());
    assert_eq!(ledger.phase, WorkPhase::Ready);
}
#[test]
fn independent_documents_sum_every_axis_without_context_concatenation() {
    let mut ledger = WorkLedger::new(twice());
    ledger.begin(work()).unwrap(); ledger.finish(None);
    ledger.begin(work()).unwrap(); ledger.finish(None);
    assert_eq!(ledger.spent, twice());
    assert_eq!(ledger.spent.attention_pairs, 2 * work().attention_pairs);
    assert_eq!(ledger.spent.projections, work().projections.checked_add(work().projections).unwrap());
    let error = ledger.begin(work()).unwrap_err();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
    assert_eq!(ledger.spent, twice());
}
#[test]
fn recoverable_result_refusal_never_refunds_the_native_attempt() {
    let mut ledger = WorkLedger::new(twice());
    ledger.begin(work()).unwrap();
    ledger.finish(Some(&BatchItemFailure::reject(BatchCode::OutputLineLimit)));
    assert_eq!(ledger.spent, work());
    ledger.begin(work()).unwrap(); ledger.finish(None);
    assert_eq!(ledger.spent, twice());
    assert!(ledger.preview(work()).is_err());
}
#[test]
fn fatal_native_or_host_failure_permanently_stops_admission() {
    let mut ledger = WorkLedger::new(twice());
    ledger.begin(work()).unwrap();
    ledger.finish(Some(&BatchItemFailure::fatal(BatchCode::Admission)));
    assert_eq!(ledger.phase, WorkPhase::Failed);
    let error = ledger.begin(work()).unwrap_err();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    assert_eq!(ledger.spent, work());
}
#[test]
fn interrupted_running_attempt_cannot_be_reentered_or_refunded() {
    let mut ledger = WorkLedger::new(twice());
    ledger.begin(work()).unwrap();
    assert!(ledger.ready().unwrap_err().stop);
    assert!(ledger.begin(work()).unwrap_err().stop);
    assert_eq!(ledger.spent, work());
    assert_eq!(ledger.phase, WorkPhase::Running);
}
#[test]
fn each_of_five_counters_can_independently_refuse_an_attempt() {
    for axis in 0..5 {
        let mut ceiling = work();
        match axis { 0 => ceiling.forward_positions -= 1, 1 => ceiling.projected_logits -= 1,
            2 => ceiling.attention_pairs -= 1, 3 => ceiling.projections.dot_products -= 1,
            _ => ceiling.projections.multiply_accumulates -= 1 }
        let mut ledger = WorkLedger::new(ceiling);
        let error = ledger.begin(work()).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert!(!error.stop); assert_eq!(ledger.spent, Int8Work::default());
        assert_eq!(ledger.phase, WorkPhase::Ready);
    }
}
#[test]
fn arithmetic_overflow_refuses_without_partial_ledger_update() {
    let maximum = Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
        attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } };
    for axis in 0..5 {
        let mut ledger = WorkLedger::new(maximum);
        match axis { 0 => ledger.spent.forward_positions = u64::MAX, 1 => ledger.spent.projected_logits = u64::MAX,
            2 => ledger.spent.attention_pairs = u64::MAX, 3 => ledger.spent.projections.dot_products = u64::MAX,
            _ => ledger.spent.projections.multiply_accumulates = u64::MAX }
        let before = ledger.spent;
        assert_eq!(ledger.begin(work()).unwrap_err().fault.code, BatchCode::WorkLimit);
        assert_eq!(ledger.spent, before); assert_eq!(ledger.phase, WorkPhase::Ready);
    }
}
#[test]
fn bogus_completion_cannot_reset_a_failed_or_unstarted_ledger() {
    let mut ledger = WorkLedger::new(twice());
    ledger.finish(None);
    assert_eq!(ledger.phase, WorkPhase::Failed);
    ledger.finish(None);
    assert!(ledger.ready().unwrap_err().stop);
}
#[test]
fn exact_pinned_planner_binding_is_required_before_model_allocation() {
    let id = identity();
    check_binding(&id, &id.template_digest, id.tokenizer_digest).unwrap();
    let other = Sha256Digest::of_bytes(b"other");
    assert!(check_binding(&id, &other, id.tokenizer_digest).is_err());
    assert!(check_binding(&id, &id.template_digest, other).is_err());
}
#[test]
fn eager_wrong_task_backend_or_kv_cannot_be_silently_repaired() {
    for axis in 0..5 {
        let mut id = identity();
        match axis { 0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.task_spec = "classify-v1".to_owned(), 2 => id.backend_semantic_version.push('x'),
            3 => id.kv_dtype = "int8".to_owned(), _ => id.schema_version += 1 }
        assert!(check_binding(&id, &id.template_digest, id.tokenizer_digest).is_err());
    }
}
#[test]
fn user_planning_refusals_are_recoverable_but_execution_contract_errors_are_not() {
    for error in [Int8JudgeError::Task(JudgeError::Contract("empty input")),
        Int8JudgeError::Task(JudgeError::Limit("prompt_tokens"))] {
        assert!(!planning_failure(error.clone()).stop);
        assert!(execution_failure(error).stop);
    }
    assert!(!planning_failure(Int8JudgeError::Native(StrictInt8Error::Context)).stop);
    assert!(!planning_failure(Int8JudgeError::Native(StrictInt8Error::Memory)).stop);
}
#[test]
fn allocation_serialization_and_identity_failures_are_terminal() {
    for (error, code) in [
        (Int8JudgeError::Task(JudgeError::AllocationRefused), BatchCode::Allocation),
        (Int8JudgeError::Native(StrictInt8Error::Allocation), BatchCode::Allocation),
        (Int8JudgeError::Scoring(Int8ScoringError::Allocation), BatchCode::Allocation),
        (Int8JudgeError::Task(JudgeError::Serialization), BatchCode::Serialization),
        (Int8JudgeError::Identity, BatchCode::Admission),
        (Int8JudgeError::Accounting, BatchCode::InvalidExecution),
    ] {
        let failure = planning_failure(error);
        assert!(failure.stop); assert_eq!(failure.fault.code, code);
    }
}
#[test]
fn native_cancellation_cause_survives_even_a_poisoned_engine() {
    for healthy in [true, false] {
        let failure = execution_outcome_failure(Int8JudgeError::Scoring(Int8ScoringError::Native(
            StrictInt8Error::Cancelled(DecodeCancellationKind::CostBudget))), healthy);
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Cancelled);
        assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::CostBudget));
    }
}
#[test]
fn output_refusal_only_allows_continuation_with_a_clean_native_engine() {
    for healthy in [true, false] {
        let failure = execution_outcome_failure(Int8JudgeError::Task(JudgeError::Limit("complete_output_bytes")), healthy);
        assert_eq!(failure.stop, !healthy);
        assert_eq!(failure.fault.code, BatchCode::OutputLineLimit);
    }
}
#[test]
fn common_batch_arguments_preserve_exact_text_in_all_three_modes() {
    let text = "café\n<|im_start|>system";
    let pairwise = JudgeBatchArgs::Pairwise { criterion: "Accuracy".to_owned(), b: "B".to_owned(),
        policy: PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 20000 }, budget: budget() };
    let JudgeRequest::Pairwise { a, b, .. } = pairwise.into_request(text.to_owned()) else { panic!("pairwise") };
    assert_eq!(a, text); assert_eq!(b, "B");
    let rubric = JudgeBatchArgs::Rubric { rubric: RubricDefinition { schema_version: 1, revision: "local".to_owned(),
        declared_origin_digest: Sha256Digest::of_bytes(b"local"), scale_maximum: 1, criteria: vec![] },
        policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1000000 }, budget: budget() };
    let JudgeRequest::Rubric { document, .. } = rubric.into_request(text.to_owned()) else { panic!("rubric") };
    assert_eq!(document, text); // Shape conversion does not replace real planner validation.
    let faithfulness = JudgeBatchArgs::Faithfulness { claim: "original claim".to_owned(),
        policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 900000, minimum_margin_milli: 100,
            evidence_window_bytes: 64, max_evidence_windows: 8, max_evidence_spans: 8 }, budget: budget() };
    let JudgeRequest::Faithfulness { source, claim, .. } = faithfulness.into_request(text.to_owned()) else { panic!("faithfulness") };
    assert_eq!(source, text); assert_eq!(claim, "original claim");
}
#[test]
fn owned_corpus_configuration_and_io_cross_the_real_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<JudgeCorpusConfig>();
    send::<StreamInput<(Arc<JudgePlanner>, JudgeCorpusConfig), std::io::Cursor<Vec<u8>>, Vec<u8>>>();
}
