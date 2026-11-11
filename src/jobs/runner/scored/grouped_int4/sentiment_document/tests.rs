//! Model-free recipe and durable-storage regressions. These exercise real
//! commitments and journals, not native scoring, task quality or DSR evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::{Int4MemoryRequirement, STRICT_INT4_EXECUTION}},
    tasks::sentiment::SentimentAxis,
};

mod storage;

const WIDTH: usize = 32;
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(1024, WIDTH).unwrap().kv_bytes * 4 }
}
fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: 10_000 * n, projected_logits: 1_000_000 * n,
        attention_pairs: 10_000_000 * n,
        projections: ProjectionWork { dot_products: 1_000_000_000 * n,
            multiply_accumulates: 10_000_000_000 * n } }
}
fn settings() -> SentimentBatchArgs {
    SentimentBatchArgs { axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget: budget() }
}
fn mapping() -> Int4SentimentMapLimits {
    Int4SentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 32,
        max_chunk_tokens: 64, context_tokens: 1024, reserved_tokens: 16,
        max_chunks: 64, max_tokenizer_calls: 4096 }, max_model_work: work(3),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 2 << 20,
        aggregation: SentimentAggregationPolicy::default() }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 65_536, max_result_bytes: 4 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(work(6), 0) }
}
fn recipe(map: Int4SentimentMapLimits, settings: SentimentBatchArgs,
    planning: SentimentLimits, limits: JobLimits) -> Int4SentimentDocumentJobRecipe {
    // Synthetic model metadata cannot construct a public native planner. Only
    // its actual private serialization and job binding are exercised here.
    let sentiment = Int4ScoredJobRecipe::sentiment(WIDTH, budget(), planning,
        Some(settings), Int4ScoredBatchLimits { max_item_work: map.max_model_work,
            max_model_work: limits.max_work.int4_model_work() }).unwrap();
    Int4SentimentDocumentJobRecipe::new(sentiment, map).unwrap()
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"sentiment document job fixture, not native model evidence");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, WIDTH)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"), packing_set_digest: packing,
        tokenizer_digest: digest, template_digest: digest, task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: digest,
        decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn freeze(recipe: &Int4SentimentDocumentJobRecipe) -> FrozenManifest {
    freeze_source(recipe, br#"{"id":"document","text":"private complete source with a final tail"}"#)
}
fn freeze_source(recipe: &Int4SentimentDocumentJobRecipe, source: &[u8]) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: &identity(), recipe, limits: limits() },
        [JobInput { id: "document", original: source, normalized: source }], &mut Continue).unwrap()
}
fn axis(work: &mut Int4Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn every_partition_native_work_and_aggregation_limit_authenticates_resume() {
    let original = recipe(mapping(), settings(), SentimentLimits::default(), limits());
    for field in 0..15 {
        let mut changed = mapping();
        match field {
            0 => changed.chunks.max_input_bytes += 1,
            1 => changed.chunks.max_chunk_bytes += 1,
            2 => changed.chunks.max_chunk_tokens += 1,
            3 => changed.chunks.context_tokens += 1,
            4 => changed.chunks.reserved_tokens += 1,
            5 => changed.chunks.max_chunks += 1,
            6 => changed.chunks.max_tokenizer_calls += 1,
            7..=11 => *axis(&mut changed.max_model_work, field - 7) += 1,
            12 => changed.max_retained_result_bytes += 1,
            13 => changed.max_result_bytes += 1,
            _ => changed.aggregation.minimum_estimated_coverage_ppm -= 1,
        }
        let changed = recipe(changed, settings(), SentimentLimits::default(), limits());
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
}

#[test]
fn dimensions_task_budget_and_all_planning_counters_authenticate_resume() {
    let original = recipe(mapping(), settings(), SentimentLimits::default(), limits());
    for field in 0..7 {
        let mut changed = settings();
        match field {
            0 => changed.axes.reverse(),
            1 => changed.axes[1] = SentimentAxis::Dominance,
            2 => changed.budget.max_input_tokens -= 1,
            3 => changed.budget.max_output_tokens -= 1,
            4 => changed.budget.max_output_bytes -= 1,
            5 => changed.budget.max_grammar_states -= 1,
            _ => changed.budget.max_kv_bytes -= 1,
        }
        let changed = recipe(mapping(), changed, SentimentLimits::default(), limits());
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "dimension/budget {field}");
    }
    for field in 0..11 {
        let mut changed = SentimentLimits::default();
        match field {
            0 => changed.per_axis.max_candidates -= 1,
            1 => changed.per_axis.max_total_tokens -= 1,
            2 => changed.per_axis.max_nodes -= 1,
            3 => changed.per_axis.max_depth -= 1,
            4 => changed.per_axis.max_candidate_id_bytes -= 1,
            5 => changed.per_axis.max_projected_logits -= 1,
            6 => changed.max_total_prompt_tokens -= 1,
            7 => changed.max_total_candidate_tokens -= 1,
            8 => changed.max_total_nodes -= 1,
            9 => changed.max_total_projected_logits -= 1,
            _ => changed.max_output_bytes -= 1,
        }
        let changed = recipe(mapping(), settings(), changed, limits());
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "planning {field}");
    }
}

#[test]
fn execution_schedule_profiles_and_lifetime_debits_are_not_resume_mutable() {
    let make = || recipe(mapping(), settings(), SentimentLimits::default(), limits());
    let original = make();
    for field in 0..8 {
        let mut changed = make();
        match field {
            0 => changed.execution = INT4_SENTIMENT_MAP_COHORT_EXECUTION,
            1 => changed.cohort_chunks = Some(2),
            2 => changed.max_stored_result_bytes += 4096,
            3 => changed.chunk_profile = "different-chunk-profile",
            4 => changed.aggregation_profile = "different-aggregation-profile",
            5 => changed.reservation_policy = "different-reservation-policy",
            6 => changed.input_policy = "different-input-policy",
            _ => changed.version += 1,
        }
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
    for field in 0..5 {
        let mut changed = limits(); let mut native = changed.max_work.int4_model_work();
        *axis(&mut native, field) -= 1; changed.max_work = JobWork::from_int4(native, 0);
        let changed = recipe(mapping(), settings(), SentimentLimits::default(), changed);
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "lifetime axis {field}");
    }
}

#[test]
fn complete_source_tail_is_authenticated_without_retaining_source_in_metadata() {
    let recipe = recipe(mapping(), settings(), SentimentLimits::default(), limits());
    let original = freeze(&recipe);
    let changed = freeze_source(&recipe,
        br#"{"id":"document","text":"private complete source with a changed tail"}"#);
    assert_eq!(original.binding.compare(&changed.binding), Err(JobError::Mismatch(MismatchField::Population)));
    let retained = String::from_utf8(canonjson::canonical_bytes(&(&original.binding, &original.items)).unwrap()).unwrap();
    for private in ["private complete source", "final tail", "valence", "arousal"] { assert!(!retained.contains(private)); }
}

#[test]
fn every_native_axis_must_fit_durable_authority_and_input_output_are_bounded() {
    check_limits(mapping(), limits()).unwrap();
    for field in 0..5 {
        let mut limited = limits(); let mut maximum = mapping().max_model_work;
        *axis(&mut maximum, field) -= 1; limited.max_work = JobWork::from_int4(maximum, 0);
        assert_eq!(check_limits(mapping(), limited), Err(BatchCode::InvalidLimits.into()), "field {field}");
    }
    let mut masks = limits(); masks.max_work.mask_node_visits = 1;
    assert!(check_limits(mapping(), masks).is_err());
    let mut output = limits(); output.max_result_bytes = mapping().max_result_bytes as usize - 1;
    assert!(check_limits(mapping(), output).is_err());
    let mut input = limits(); input.max_input_bytes_per_item = mapping().chunks.max_input_bytes - 1;
    assert!(check_limits(mapping(), input).is_err());
}

#[test]
fn record_arguments_cannot_override_fixed_axes_budget_or_map_policy() {
    for input in [r#"{"id":"a","text":"source"}"#,
        r#"{"id":"a","text":"source","task_args":null}"#,
        r#"{"id":"a","text":"source","task_args":{}}"#] {
        let parsed = serde_json::from_str::<BatchDocument<Int4SentimentDocumentArgs>>(input).unwrap();
        let fixed = fixed_document(parsed);
        assert_eq!(fixed.id, "a"); assert_eq!(fixed.text, "source"); assert!(fixed.task_args.is_none());
    }
    for value in [r#"{"axes":[]}"#, r#"{"axes":["approach"]}"#, r#"{"budget":{}}"#,
        r#"{"max_chunks":1}"#, r#"{"cohort_chunks":1}"#, r#"{"minimum_estimated_coverage_ppm":0}"#,
        "[]", "[{}]", "true", "0", r#""""#] {
        let input = format!(r#"{{"id":"a","text":"source","task_args":{value}}}"#);
        assert!(serde_json::from_str::<BatchDocument<Int4SentimentDocumentArgs>>(&input).is_err(), "{value}");
    }
}

#[test]
fn original_axis_capacity_remains_visible_to_host_admission() {
    let mut settings = settings(); settings.axes.reserve_exact(4096);
    let allocation = (settings.axes.as_ptr(), settings.axes.capacity());
    let expected = serde_json::to_value(&settings).unwrap();
    let native = Int4ScoredBatchLimits { max_item_work: mapping().max_model_work,
        max_model_work: limits().max_work.int4_model_work() };
    let (recipe, retained) = Int4ScoredJobRecipe::sentiment_retaining_defaults(
        WIDTH, budget(), SentimentLimits::default(), settings, native).unwrap();
    assert_eq!((retained.axes.as_ptr(), retained.axes.capacity()), allocation);
    assert_eq!(serde_json::to_value(&retained).unwrap(), expected);
    assert_eq!(serde_json::to_value(recipe).unwrap()["settings"]["defaults"], expected);
    for axes in [Vec::new(), vec![SentimentAxis::Valence; 2]] {
        assert!(Int4ScoredJobRecipe::sentiment_retaining_defaults(WIDTH, budget(),
            SentimentLimits::default(), SentimentBatchArgs { axes, budget: budget() }, native).is_err());
    }
}

fn compact_policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 262_144, max_output_tokens: 131_072,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28,
    })
}

#[test]
fn compact_policy_and_every_tokenizer_ceiling_authenticate_resume_without_changing_byte_recipe() {
    let make = || recipe(mapping(), settings(), SentimentLimits::default(), limits());
    let byte = make();
    assert!(serde_json::to_value(&byte).unwrap().get("input_encoding").is_none());
    let mut compact = make(); compact.input_encoding = Some(compact_policy());
    assert_eq!(freeze(&byte).binding.compare(&freeze(&compact).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    for axis in 0..4 {
        let mut changed = make();
        let UntrustedEncoding::SafeBpeV1(mut policy) = compact_policy() else { panic!("compact fixture"); };
        match axis { 0 => policy.max_input_bytes -= 1, 1 => policy.max_output_tokens -= 1,
            2 => policy.max_pair_checks -= 1, _ => policy.max_pair_bytes -= 1 }
        changed.input_encoding = Some(UntrustedEncoding::SafeBpeV1(policy));
        assert_eq!(freeze(&compact).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
    }
    let mut cohorts = make(); cohorts.input_encoding = Some(compact_policy());
    cohorts.execution = INT4_SENTIMENT_MAP_COHORT_EXECUTION; cohorts.cohort_chunks = Some(2);
    cohorts.max_stored_result_bytes += 4096;
    assert_eq!(freeze(&compact).binding.compare(&freeze(&cohorts).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(freeze(&compact).binding.compare(&freeze(&compact).binding), Ok(()));
}
