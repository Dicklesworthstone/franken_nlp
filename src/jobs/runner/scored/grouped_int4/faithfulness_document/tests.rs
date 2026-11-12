//! Real keyed recipe and population contracts with synthetic model metadata.
//! These checks neither execute a substitute judge nor qualify native scores.
use super::*;
use crate::{artifact::converter::PINNED_REVISION, canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::{Int4MemoryRequirement, STRICT_INT4_EXECUTION}}};

mod storage;

fn compact_policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 1 << 18, max_output_tokens: 1 << 17,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 })
}

const WIDTH: usize = 32;
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(1024, WIDTH).unwrap().kv_bytes * 32 }
}
fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: 10_000 * n, projected_logits: 1_000_000 * n,
        attention_pairs: 10_000_000 * n, projections: ProjectionWork { dot_products: 1_000_000_000 * n,
            multiply_accumulates: 10_000_000_000 * n } }
}
fn policy() -> FaithfulnessPolicy {
    FaithfulnessPolicy { minimum_candidate_weight_ppm: 700_000, minimum_margin_milli: 200,
        evidence_window_bytes: 16, max_evidence_windows: 31, max_evidence_spans: 30 }
}
fn mapping() -> Int4FaithfulnessMapLimits {
    Int4FaithfulnessMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 32,
        max_chunk_tokens: 64, context_tokens: 1024, reserved_tokens: 16,
        max_chunks: 64, max_tokenizer_calls: 4096 }, max_model_work: work(3),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 2 << 20,
        aggregation: FaithfulnessAggregationPolicy::default() }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 65_536, max_result_bytes: 4 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(work(6), 0) }
}
fn recipe(map: Int4FaithfulnessMapLimits, policy: FaithfulnessPolicy, task: TaskBudget,
    planning: JudgeLimits, limits: JobLimits) -> Int4FaithfulnessDocumentJobRecipe {
    let judge = Int4ScoredJobRecipe::judge(WIDTH, task, planning, None,
        Int4ScoredBatchLimits { max_item_work: map.max_model_work,
            max_model_work: limits.max_work.int4_model_work() }).unwrap();
    Int4FaithfulnessDocumentJobRecipe::new(judge, policy, map).unwrap()
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"faithfulness document job storage fixture, not native evidence");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, WIDTH)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"), packing_set_digest: packing,
        tokenizer_digest: digest, template_digest: digest, task_spec: "judge-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: digest,
        decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn freeze(recipe: &Int4FaithfulnessDocumentJobRecipe) -> FrozenManifest {
    freeze_source(recipe, br#"{"id":"document","text":"private complete source tail","task_args":{"claim":"private exact claim"}}"#)
}
fn freeze_source(recipe: &Int4FaithfulnessDocumentJobRecipe, source: &[u8]) -> FrozenManifest {
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
fn every_complete_source_limit_and_evidence_policy_authenticates_resume() {
    let original = recipe(mapping(), policy(), budget(), JudgeLimits::default(), limits());
    for field in 0..20 {
        let mut map = mapping(); let mut policy = policy();
        match field {
            0 => map.chunks.max_input_bytes += 1, 1 => map.chunks.max_chunk_bytes += 1,
            2 => map.chunks.max_chunk_tokens += 1, 3 => map.chunks.context_tokens += 1,
            4 => map.chunks.reserved_tokens += 1, 5 => map.chunks.max_chunks += 1,
            6 => map.chunks.max_tokenizer_calls += 1,
            7..=11 => *axis(&mut map.max_model_work, field - 7) += 1,
            12 => map.max_retained_result_bytes += 1, 13 => map.max_result_bytes += 1,
            14 => map.aggregation.minimum_classified_coverage_ppm -= 1,
            15 => policy.minimum_candidate_weight_ppm -= 1, 16 => policy.minimum_margin_milli += 1,
            17 => policy.evidence_window_bytes += 1, 18 => policy.max_evidence_windows -= 1,
            _ => policy.max_evidence_spans -= 1,
        }
        let changed = recipe(map, policy, budget(), JudgeLimits::default(), limits());
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
}

#[test]
fn actual_task_and_judge_planning_limits_are_in_the_private_recipe() {
    let original = recipe(mapping(), policy(), budget(), JudgeLimits::default(), limits());
    for field in 0..15 {
        let mut task = budget(); let mut planning = JudgeLimits::default();
        match field {
            0 => task.max_input_tokens -= 1, 1 => task.max_output_tokens -= 1,
            2 => task.max_output_bytes -= 1, 3 => task.max_grammar_states -= 1,
            4 => task.max_kv_bytes -= 1, 5 => planning.per_head.max_candidates -= 1,
            6 => planning.per_head.max_total_tokens -= 1, 7 => planning.per_head.max_nodes -= 1,
            8 => planning.per_head.max_depth -= 1, 9 => planning.per_head.max_candidate_id_bytes -= 1,
            10 => planning.per_head.max_projected_logits -= 1, 11 => planning.max_total_prompt_tokens -= 1,
            12 => planning.max_total_candidate_tokens -= 1, 13 => planning.max_total_projected_logits -= 1,
            _ => planning.max_output_bytes -= 1,
        }
        let changed = recipe(mapping(), policy(), task, planning, limits());
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
}

#[test]
fn schedule_output_envelope_and_all_lifetime_axes_are_immutable() {
    let make = || recipe(mapping(), policy(), budget(), JudgeLimits::default(), limits());
    let original = make();
    for field in 0..8 {
        let mut changed = make();
        match field {
            0 => changed.execution = INT4_FAITHFULNESS_MAP_COHORT_EXECUTION,
            1 => changed.cohort_chunks = Some(22), 2 => changed.max_stored_result_bytes += 4096,
            3 => changed.chunk_profile = "different-chunk-profile",
            4 => changed.aggregation_profile = "different-evidence-profile",
            5 => changed.reservation_policy = "different-debit-policy",
            6 => changed.input_policy = "different-claim-policy", _ => changed.version += 1,
        }
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
    for field in 0..5 {
        let mut lifetime = limits(); let mut native = lifetime.max_work.int4_model_work();
        *axis(&mut native, field) -= 1; lifetime.max_work = JobWork::from_int4(native, 0);
        let changed = recipe(mapping(), policy(), budget(), JudgeLimits::default(), lifetime);
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "axis {field}");
    }
}

#[test]
fn omitted_head_cap_preserves_the_exact_legacy_private_recipe() {
    let actual = recipe(mapping(), policy(), budget(), JudgeLimits::default(), limits());
    // This is the complete wire projection before bounded head declarations
    // existed. No null field, new version, changed input policy or default
    // cap may invalidate a previously committed source/claim population.
    let legacy = serde_json::json!({
        "version": 1,
        "dependency_scope": "item-local-complete-source-claim",
        "input_policy": "fixed-faithfulness-policy-required-claim-item-args-v1",
        "execution": INT4_FAITHFULNESS_MAP_EXECUTION,
        "chunk_profile": CHUNK_PROFILE,
        "aggregation_profile": FAITHFULNESS_AGGREGATION_PROFILE,
        "reservation_policy": "exact-all-chunk-five-axis-work-before-durable-attempt-v1",
        "judge": &actual.judge,
        "policy": policy(),
        "chunks": mapping().chunks,
        "per_document_model_work": mapping().max_model_work,
        "max_retained_result_bytes": mapping().max_retained_result_bytes,
        "max_result_bytes": mapping().max_result_bytes,
        "max_stored_result_bytes": mapping().max_result_bytes,
        "aggregation": mapping().aggregation,
        "cohort_chunks": null,
    });
    assert!(actual.max_heads.is_none());
    assert!(actual.input_encoding.is_none());
    assert_eq!(canonjson::canonical_bytes(&actual).unwrap(), canonjson::canonical_bytes(&legacy).unwrap());
}

#[test]
fn opted_in_head_cap_is_resume_authority_even_when_task_budget_stays_the_same() {
    let mut policy = policy(); policy.max_evidence_windows = 1; policy.max_evidence_spans = 1;
    let make = || recipe(mapping(), policy, budget(), JudgeLimits::default(), limits());
    let original = make(); let original_binding = freeze(&original).binding;
    for cap in [1, 3, 32] {
        let mut explicit = make(); explicit.max_heads = Some(cap);
        assert_eq!(original_binding.compare(&freeze(&explicit).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "explicit cap {cap}");
        let serialized = serde_json::to_value(&explicit).unwrap();
        assert_eq!(serialized["max_heads"], cap);
        assert_eq!(canonjson::canonical_bytes(&explicit.judge).unwrap(),
            canonjson::canonical_bytes(&original.judge).unwrap());
        let mut changed = make(); changed.max_heads = Some(if cap == 1 { 3 } else { 1 });
        assert_eq!(freeze(&explicit).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}

#[test]
fn both_exact_claim_and_final_source_tail_authenticate_without_metadata_disclosure() {
    let recipe = recipe(mapping(), policy(), budget(), JudgeLimits::default(), limits());
    let original = freeze(&recipe);
    for bytes in [
        br#"{"id":"document","text":"private complete source changed tail","task_args":{"claim":"private exact claim"}}"#.as_slice(),
        br#"{"id":"document","text":"private complete source tail","task_args":{"claim":"private changed claim"}}"#.as_slice(),
    ] {
        assert_eq!(original.binding.compare(&freeze_source(&recipe, bytes).binding),
            Err(JobError::Mismatch(MismatchField::Population)));
    }
    let metadata = String::from_utf8(canonjson::canonical_bytes(&(&original.binding, &original.items)).unwrap()).unwrap();
    for private in ["private complete source", "exact claim", "evidence_window_bytes", "minimum_margin_milli"] {
        assert!(!metadata.contains(private));
    }
}

#[test]
fn source_claim_output_and_each_work_axis_must_fit_stored_authority() {
    check_limits(mapping(), limits()).unwrap();
    for field in 0..5 {
        let mut limited = limits(); let mut maximum = mapping().max_model_work;
        *axis(&mut maximum, field) -= 1; limited.max_work = JobWork::from_int4(maximum, 0);
        assert_eq!(check_limits(mapping(), limited), Err(BatchCode::InvalidLimits.into()), "axis {field}");
    }
    let mut masks = limits(); masks.max_work.mask_node_visits = 1;
    assert!(check_limits(mapping(), masks).is_err());
    let mut output = limits(); output.max_result_bytes = mapping().max_result_bytes as usize - 1;
    assert!(check_limits(mapping(), output).is_err());
    let mut input = limits(); input.max_input_bytes_per_item = mapping().chunks.max_input_bytes - 1;
    assert!(check_limits(mapping(), input).is_err());
    let mut policy = policy(); policy.max_evidence_windows = 0;
    let judge = Int4ScoredJobRecipe::judge(WIDTH, budget(), JudgeLimits::default(), None,
        Int4ScoredBatchLimits { max_item_work: mapping().max_model_work, max_model_work: work(6) }).unwrap();
    assert!(Int4FaithfulnessDocumentJobRecipe::new(judge, policy, mapping()).is_err());
}


#[test]
fn compact_tokenizer_policy_is_complete_resume_authority_without_changing_native_limits() {
    let make = || recipe(mapping(), policy(), budget(), JudgeLimits::default(), limits());
    let legacy = make();
    let mut original = make(); original.input_encoding = Some(compact_policy());
    let original_binding = freeze(&original).binding;
    assert_eq!(original_binding.compare(&freeze(&legacy).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    for axis in 0..4 {
        let mut changed = make();
        let UntrustedEncoding::SafeBpeV1(mut encoding) = compact_policy() else { panic!("compact policy required"); };
        match axis { 0 => encoding.max_input_bytes -= 1, 1 => encoding.max_output_tokens -= 1,
            2 => encoding.max_pair_checks -= 1, _ => encoding.max_pair_bytes -= 1 }
        changed.input_encoding = Some(UntrustedEncoding::SafeBpeV1(encoding));
        assert_eq!(original_binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
        assert_eq!(canonjson::canonical_bytes(&changed.judge).unwrap(),
            canonjson::canonical_bytes(&legacy.judge).unwrap());
        assert_eq!(changed.chunks, legacy.chunks);
        assert_eq!(changed.per_document_model_work, legacy.per_document_model_work);
    }
    let mut same = make(); same.input_encoding = Some(compact_policy());
    assert_eq!(original_binding.compare(&freeze(&same).binding), Ok(()));
    let wire = serde_json::to_value(&original).unwrap();
    assert_eq!(wire["input_encoding"], serde_json::to_value(compact_policy()).unwrap());
    let private = String::from_utf8(canonjson::canonical_bytes(&original_binding).unwrap()).unwrap();
    for field in ["input_encoding", "max_pair_checks", "max_pair_bytes"] { assert!(!private.contains(field)); }
}

#[test]
fn actual_model_bound_factory_projects_compact_policy_and_leaves_default_recipe_unchanged() {
    let (weights, _) = crate::native_engine::strict_int4::weights::projection_fixture(WIDTH);
    for encoding in [UntrustedEncoding::ByteFallback, compact_policy()] {
        let planner = Int4JudgePlanner::for_model(weights.identity(), budget(), JudgeLimits::default()).unwrap()
            .with_document_encoding(encoding.clone()).unwrap();
        let planner = Int4FaithfulnessDocumentJobPlanner::new(planner, policy(), mapping(), limits()).unwrap();
        assert_eq!(planner.document_encoding(), &encoding);
        match encoding {
            UntrustedEncoding::ByteFallback => {
                assert!(planner.job_recipe().input_encoding.is_none());
                assert_eq!(canonjson::canonical_bytes(planner.job_recipe()).unwrap(),
                    canonjson::canonical_bytes(&recipe(mapping(), policy(), budget(), JudgeLimits::default(), limits())).unwrap());
            }
            selected => assert_eq!(planner.job_recipe().input_encoding.as_ref(), Some(&selected)),
        }
    }
}
