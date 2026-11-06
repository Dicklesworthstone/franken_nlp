//! Unrun recipe/admission/storage regressions. These are not native-model,
//! DSR, task-quality or host-performance evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::classify::{ClassificationLabel, ClassificationPolicy, ClassificationPlanningError},
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
        max_kv_bytes: Int4MemoryRequirement::for_context(1024, WIDTH).unwrap().kv_bytes * 2 }
}
fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: 10_000 * n, projected_logits: 1_000_000 * n,
        attention_pairs: 10_000_000 * n,
        projections: ProjectionWork { dot_products: 1_000_000_000 * n,
            multiply_accumulates: 10_000_000_000 * n } }
}
fn planning() -> ClassificationLimits {
    ClassificationLimits { max_labels: 2, max_input_bytes: 4096, max_context_tokens: 1024,
        ..ClassificationLimits::default() }
}
fn settings() -> ClassificationBatchArgs {
    ClassificationBatchArgs { labels: vec![
        ClassificationLabel { id: "billing".to_owned(), description: "Private invoice category é".to_owned() },
        ClassificationLabel { id: "support".to_owned(), description: "Private support category 上海".to_owned() }],
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: budget() }
}
fn mapping() -> Int4ClassificationMapLimits {
    Int4ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 32,
        max_chunk_tokens: 64, context_tokens: 1024, reserved_tokens: 16,
        max_chunks: 64, max_tokenizer_calls: 4096 }, max_model_work: work(3),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 2 << 20,
        aggregation: ClassificationAggregationPolicy::default() }
}
fn memory() -> Int4ClassificationMapMemoryLimits {
    Int4ClassificationMapMemoryLimits { context_tokens: 1024,
        preparation_reserve_bytes: 128 * 1024 * 1024, allocator_reserve_bytes: 16 * 1024 * 1024 }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 65_536, max_result_bytes: 4 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(work(6), 0) }
}
fn recipe(map: Int4ClassificationMapLimits, settings: ClassificationBatchArgs,
    planning: ClassificationLimits, limits: JobLimits) -> Int4ClassificationDocumentJobRecipe {
    // Real private projections and HMAC binding, with synthetic model metadata
    // only. This cannot construct the public native planner or execute scores.
    let classification = Int4ScoredJobRecipe::classification(WIDTH, budget(), planning,
        Some(settings), Int4ScoredBatchLimits { max_item_work: map.max_model_work,
            max_model_work: limits.max_work.int4_model_work() }).unwrap();
    Int4ClassificationDocumentJobRecipe::new(classification, map).unwrap()
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"classification document job storage fixture, not model evidence");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, WIDTH)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"), packing_set_digest: packing,
        tokenizer_digest: digest, template_digest: digest, task_spec: "classify-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: digest,
        decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn freeze(recipe: &Int4ClassificationDocumentJobRecipe) -> FrozenManifest {
    let input = br#"{"id":"document","text":"private complete source"}"#;
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: &identity(), recipe, limits: limits() },
        [JobInput { id: "document", original: input, normalized: input }], &mut Continue).unwrap()
}
fn axis(work: &mut Int4Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn every_document_partition_work_result_and_aggregation_field_changes_resume_binding() {
    let original = recipe(mapping(), settings(), planning(), limits());
    for field in 0..16 {
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
            14 => changed.aggregation.minimum_classified_coverage_ppm -= 1,
            _ => changed.chunks.max_chunk_bytes -= 1,
        }
        let changed = recipe(changed, settings(), planning(), limits());
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
}

#[test]
fn fixed_private_labels_mode_local_policy_and_task_budget_remain_immutable() {
    let original = recipe(mapping(), settings(), planning(), limits());
    for field in 0..11 {
        let mut changed = settings();
        match field {
            0 => changed.labels[0].id.push('x'),
            1 => changed.labels[0].description.push('x'),
            2 => changed.mode = ClassificationMode::MultiLabel,
            3 => changed.policy.minimum_candidate_weight_ppm += 1,
            4 => changed.policy.minimum_margin_ppm += 1,
            5 => changed.budget.max_input_tokens -= 1,
            6 => changed.budget.max_output_tokens -= 1,
            7 => changed.budget.max_output_bytes -= 1,
            8 => changed.budget.max_grammar_states -= 1,
            9 => changed.budget.max_kv_bytes -= 1,
            _ => changed.labels.reverse(),
        }
        let changed = recipe(mapping(), changed, planning(), limits());
        assert_eq!(freeze(&original).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
}

#[test]
fn native_chunk_schedule_and_width_do_not_change_silently_on_resume() {
    let serial = recipe(mapping(), settings(), planning(), limits());
    let mut grouped = recipe(mapping(), settings(), planning(), limits());
    grouped.execution = INT4_CLASSIFICATION_MAP_COHORT_EXECUTION;
    grouped.cohort_chunks = Some(2);
    grouped.max_stored_result_bytes = cohort_schedule(WIDTH, budget(), planning(), mapping(), 1, 2,
        memory(), limits()).unwrap().required.memory.max_output_bytes;
    assert_eq!(freeze(&serial).binding.compare(&freeze(&grouped).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    let two = freeze(&grouped);
    grouped.cohort_chunks = Some(3);
    assert_eq!(two.binding.compare(&freeze(&grouped).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    // No transient host reservation or kernel choice is serialized as policy.
    let value = serde_json::to_value(&grouped).unwrap();
    for key in ["kernel", "allocator_reserve_bytes", "preparation_reserve_bytes"] {
        assert!(!value.as_object().unwrap().contains_key(key));
    }
}

#[test]
fn source_population_changes_do_not_reuse_complete_document_results() {
    let r = recipe(mapping(), settings(), planning(), limits());
    let original = freeze(&r);
    let changed = br#"{"id":"document","text":"private complete source with changed tail"}"#;
    let changed = FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: &identity(), recipe: &r, limits: limits() },
        [JobInput { id: "document", original: changed, normalized: changed }], &mut Continue).unwrap();
    assert_eq!(original.binding.compare(&changed.binding), Err(JobError::Mismatch(MismatchField::Population)));
    let retained = canonjson::canonical_bytes(&(&original.binding, &original.items)).unwrap();
    let retained = String::from_utf8(retained).unwrap();
    for private in ["private complete source", "billing", "Private invoice", "上海"] { assert!(!retained.contains(private)); }
}

#[test]
fn every_native_axis_must_fit_durable_authority_and_mask_work_is_refused() {
    validate_limits(WIDTH, budget(), budget(), planning(), mapping(), limits()).unwrap();
    for field in 0..5 {
        let mut limited = limits(); let mut maximum = mapping().max_model_work;
        *axis(&mut maximum, field) -= 1;
        limited.max_work = JobWork::from_int4(maximum, 0);
        assert_eq!(validate_limits(WIDTH, budget(), budget(), planning(), mapping(), limited),
            Err(BatchCode::InvalidLimits.into()), "field {field}");
    }
    let mut masks = limits(); masks.max_work.mask_node_visits = 1;
    assert!(validate_limits(WIDTH, budget(), budget(), planning(), mapping(), masks).is_err());
    let mut output = limits(); output.max_result_bytes = mapping().max_result_bytes as usize - 1;
    assert!(validate_limits(WIDTH, budget(), budget(), planning(), mapping(), output).is_err());
    let mut small = budget(); small.max_kv_bytes = 1;
    assert!(validate_limits(WIDTH, budget(), small, planning(), mapping(), limits()).is_err());
}

#[test]
fn cohort_geometry_requires_full_declared_window_and_complete_stored_output() {
    let serial = memory().requirements(WIDTH, budget(), planning(), mapping()).unwrap();
    let grouped = cohort_schedule(WIDTH, budget(), planning(), mapping(), 2, 3, memory(), limits()).unwrap();
    assert_eq!(grouped.required.memory.native.kv_bytes, serial.native.kv_bytes * 6);
    assert_eq!(grouped.required.memory.max_output_bytes, mapping().max_result_bytes + 4096);
    assert!(grouped.required.memory.input_bytes > serial.input_bytes);
    let mut too_small = limits(); too_small.max_result_bytes = mapping().max_result_bytes as usize;
    assert!(cohort_schedule(WIDTH, budget(), planning(), mapping(), 2, 3, memory(), too_small).is_err());
    assert!(cohort_schedule(WIDTH, budget(), planning(), mapping(), 2, 33, memory(), limits()).is_err());
    assert!(cohort_schedule(WIDTH, budget(), planning(), mapping(), 2, 0, memory(), limits()).is_err());
    let mut one_head = budget(); one_head.max_kv_bytes = serial.native.kv_bytes;
    assert!(cohort_schedule(WIDTH, one_head, planning(), mapping(), 2, 2, memory(), limits()).is_err());
    let mut other_context = memory(); other_context.context_tokens += 1;
    assert!(cohort_schedule(WIDTH, budget(), planning(), mapping(), 1, 2, other_context, limits()).is_err());
}

#[test]
fn record_arguments_cannot_override_frozen_task_or_mapping_policy() {
    for input in [r#"{"id":"a","text":"source"}"#,
        r#"{"id":"a","text":"source","task_args":null}"#,
        r#"{"id":"a","text":"source","task_args":{}}"#] {
        assert!(serde_json::from_str::<BatchDocument<Int4ClassificationDocumentArgs>>(input).is_ok());
    }
    for override_value in [r#"{"labels":[]}"#, r#"{"mode":"multi_label"}"#,
        r#"{"budget":{}}"#, r#"{"max_chunks":1}"#, r#"{"cohort_chunks":1}"#,
        r#"{"minimum_classified_coverage_ppm":0}"#, "[]", "[{}]", "true", "0", r#""""#] {
        let input = format!(r#"{{"id":"a","text":"source","task_args":{override_value}}}"#);
        assert!(serde_json::from_str::<BatchDocument<Int4ClassificationDocumentArgs>>(&input).is_err(), "{override_value}");
    }
}

#[test]
fn frozen_policy_keeps_original_label_allocations_visible_to_host_admission() {
    let mut settings = settings();
    settings.labels.reserve_exact(planning().max_labels * 8);
    settings.labels[0].description.reserve_exact(planning().max_total_label_bytes * 2);
    let labels = (settings.labels.as_ptr(), settings.labels.capacity());
    let description = (settings.labels[0].description.as_ptr(), settings.labels[0].description.capacity());
    let expected = serde_json::to_value(&settings).unwrap();
    let (recipe, retained) = Int4ScoredJobRecipe::classification_retaining_defaults(WIDTH,
        budget(), planning(), settings, Int4ScoredBatchLimits { max_item_work: mapping().max_model_work,
            max_model_work: limits().max_work.int4_model_work() }).unwrap();
    assert_eq!((retained.labels.as_ptr(), retained.labels.capacity()), labels);
    assert_eq!((retained.labels[0].description.as_ptr(), retained.labels[0].description.capacity()), description);
    assert_eq!(serde_json::to_value(&retained).unwrap(), expected);
    assert_eq!(serde_json::to_value(recipe).unwrap()["settings"]["defaults"], expected);
    let mut invalid = retained; invalid.labels.clear();
    assert!(Int4ScoredJobRecipe::classification_retaining_defaults(WIDTH, budget(), planning(), invalid,
        Int4ScoredBatchLimits { max_item_work: mapping().max_model_work,
            max_model_work: limits().max_work.int4_model_work() }).is_err());
}

#[test]
fn all_cancellation_kinds_remain_terminal_and_typed() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        struct Stop(DecodeCancellationKind);
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
        }
        let first = poll(&mut Stop(cause)).unwrap_err();
        assert!(first.stop); assert_eq!(first.fault, BatchFault::cancelled(cause));
        let nested = map_failure(ClassificationPlanningError::Cancelled(cause).into());
        assert!(nested.stop); assert_eq!(nested.fault, BatchFault::cancelled(cause));
    }
}

struct Admission;
impl Int4ClassificationDocumentBatchAdmission for Admission {
    type Guard = ();
    fn admit(&mut self, _: Int4ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn actual_native_processor_implements_the_guarded_durable_boundary() {
    fn durable<T: DurableBatchProcessor>() {}
    fn send<T: Send + 'static>() {}
    durable::<Int4ClassificationDocumentJobProcessor<'static, 'static, 'static, Admission>>();
    send::<Int4ClassificationDocumentJobPlanner>();
}
