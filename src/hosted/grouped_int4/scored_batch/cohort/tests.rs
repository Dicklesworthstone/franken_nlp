//! Structural admission tests; fixtures never authorize native model execution.
use super::*;
use crate::{artifact::converter::PINNED_REVISION,
    batch::{BatchLimits, scored_int4::Int4ScoredPlanFacts},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::TaskBudget, classify::grouped_int4::Int4ClassificationPlanner,
        sentiment::grouped_int4::Int4SentimentPlanner, judge::grouped_int4::Int4JudgePlanner},
};

fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"cohort-host-admission-fixture-only");
    let packing = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, 32_usize,
    )).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"), packing_set_digest: packing,
        tokenizer_digest: digest, template_digest: digest, task_spec: task.to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "complete-candidate-eos".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn work() -> Int4Work {
    Int4Work::for_sequence(32, 0, 48, 3 * NANBEIGE_VOCAB_SIZE).unwrap()
}
fn request(identity: &ExecutionIdentity) -> Int4ScoredAdmission<'_> {
    let kv = Int4MemoryRequirement::for_cohort(&[24, 32, 16], 32).unwrap().kv_bytes;
    Int4ScoredAdmission { facts: Int4ScoredPlanFacts { identity, model_work: work(),
        group_width: 32, head_count: 3, required_context: 32,
        task_budget: TaskBudget { max_input_tokens: 32, max_output_tokens: 16,
            max_output_bytes: 8192, max_grammar_states: 256, max_kv_bytes: kv },
        max_result_bytes: 8192 }, kv_reservation_bytes: kv }
}
fn check(task: &str, request: &Int4ScoredAdmission<'_>) -> Result<(), BatchItemFailure> {
    check_cohort_request(32, 32, 6,
        Int4MemoryRequirement::for_cohort(&[32; 6], 32).unwrap().kv_bytes,
        work(), task, 8192, request)
}

#[test]
fn complete_simultaneous_ragged_kv_is_admitted_for_all_three_tasks() {
    for task in ["classify-v1", "sentiment-v1", "judge-v1"] {
        let identity = identity(task);
        let request = request(&identity);
        assert!(request.kv_reservation_bytes
            > Int4MemoryRequirement::for_context(32, 32).unwrap().kv_bytes);
        check(task, &request).unwrap();
    }
}

#[test]
fn serial_kv_charge_cannot_impersonate_a_simultaneous_bundle() {
    let identity = identity("judge-v1"); let mut request = request(&identity);
    request.kv_reservation_bytes = Int4MemoryRequirement::for_context(32, 32).unwrap().kv_bytes;
    assert!(check("judge-v1", &request).unwrap_err().stop);
    let mut request = self::request(&identity);
    request.facts.task_budget.max_kv_bytes = request.kv_reservation_bytes - 1;
    assert!(check("judge-v1", &request).unwrap_err().stop);
}

#[test]
fn head_context_model_and_output_boundaries_are_not_repaired() {
    let identity = identity("sentiment-v1");
    for axis in 0..7 {
        let mut request = request(&identity);
        match axis {
            0 => request.facts.head_count = 0, 1 => request.facts.head_count = 7,
            2 => request.facts.required_context = 0, 3 => request.facts.required_context = 33,
            4 => request.facts.group_width = 128, 5 => request.kv_reservation_bytes = u64::MAX,
            _ => request.kv_reservation_bytes = 0,
        }
        assert!(check("sentiment-v1", &request).unwrap_err().stop, "axis {axis}");
    }
    assert!(check("judge-v1", &request(&identity)).unwrap_err().stop);
    for limit in [0, 8193, u64::MAX] {
        let mut request = request(&identity); request.facts.max_result_bytes = limit;
        let error = check("sentiment-v1", &request).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
        assert!(!error.stop);
    }
}

#[test]
fn all_five_work_counters_are_checked_without_a_dense_vocab_assumption() {
    let identity = identity("sentiment-v1");
    let mut request = request(&identity);
    request.facts.model_work = Int4Work::for_sequence(32, 0, 48, 5).unwrap();
    check("sentiment-v1", &request).unwrap();
    for axis in 0..5 {
        let mut ceiling = request.facts.model_work;
        match axis {
            0 => ceiling.forward_positions -= 1, 1 => ceiling.projected_logits -= 1,
            2 => ceiling.attention_pairs -= 1, 3 => ceiling.projections.dot_products -= 1,
            _ => ceiling.projections.multiply_accumulates -= 1,
        }
        let error = check_cohort_request(32, 32, 6,
            Int4MemoryRequirement::for_cohort(&[32; 6], 32).unwrap().kv_bytes,
            ceiling, "sentiment-v1", 8192, &request).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn host_and_unconditional_cli_quote_use_exactly_the_same_reservations() {
    let cohort = Int4ScoredCohortHostLimits { max_documents: 3, max_heads: 6 };
    let limits = CorpusLimits { native: NativeLimits { context_tokens: 32,
        allocator_reserve_bytes: 1 << 20, run: RunLimits {
            max_elapsed: Duration::from_secs(30), max_checkpoints: 1000, cleanup_reserve_bytes: 65_536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 8 << 20, io_reserve_bytes: 65_536 };
    let work = Int4ScoredBatchLimits { max_item_work: work(), max_model_work: work() };
    let expected = Int4ScoredCohortMemoryLimits { max_documents: cohort.max_documents,
        max_heads: cohort.max_heads, context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
    }.requirements(32, work, limits.transport).unwrap();
    assert_eq!(cohort.requirements(32, work, limits).unwrap(), expected);
    let mut invalid = limits; invalid.native.run.max_checkpoints = 0;
    assert!(cohort.requirements(32, work, invalid).is_err());
}

#[test]
fn owned_io_and_all_three_sealed_planners_cross_one_host_entrypoint() {
    let _classify = NlpEngine::batch_int4_scored_cohorts::<Int4ClassificationPlanner, std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _sentiment = NlpEngine::batch_int4_scored_cohorts::<Int4SentimentPlanner, std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _judge = NlpEngine::batch_int4_scored_cohorts::<Int4JudgePlanner, std::io::Cursor<Vec<u8>>, Vec<u8>>;
    fn send<T: Send + 'static>() {}
    send::<Int4ScoredCohortBatchResult>();
}
