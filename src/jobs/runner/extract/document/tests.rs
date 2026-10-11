//! Model-free frozen-contract and real pinned planning regressions.
//! No fixture constructs a native engine or claims successful model execution.
use super::*;
use crate::{
    batch::extract::ExtractionBatchGrounding,
    canonjson, execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField,
        runner::JobRunError},
    native_engine::{constrained_int8, decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tokenizer::pinned_controls,
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 32, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"durable-document-extraction-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn four(w: Int8Work) -> Int8Work { let two = w.checked_add(w).unwrap(); two.checked_add(two).unwrap() }
fn config() -> ExtractionDocumentBatchConfig {
    let per = four(constrained_int8::planned_work(2016, 32).unwrap());
    ExtractionDocumentBatchConfig { request: ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::Structural, budget: budget() },
        limits: Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 2048,
                context_tokens: 2048, reserved_tokens: 32, max_chunks: 4, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { max_value_bytes: 262144, max_live_value_bytes: 1 << 20,
                max_total_value_bytes: 4 << 20, max_result_bytes: 262144, ..ExecutionLimits::default() },
            max_model_work: per, mask_limits: MaskWorkLimits::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 4000 },
            verification: GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 10000 } },
        max_model_work: four(per), max_mask_visits: 16000,
        max_verification: GroundingBudget { max_fields: 128, max_matches: 512, max_scan_steps: 40000 } }
}
fn factory(c: ExtractionDocumentBatchConfig) -> Int8DocumentExtractionJobPlanner {
    let registry = pinned_controls::pinned().unwrap();
    Int8DocumentExtractionJobPlanner::pinned(registry.template_controls(), 166101, identity(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), c).unwrap()
}
fn limits() -> JobLimits {
    let c = config();
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192, max_snapshot_bytes: 65536,
        max_result_bytes: 262144, max_spool_bytes: 16 << 20, max_materialized_bytes: 16 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: c.max_model_work, mask_node_visits: c.max_mask_visits } }
}
fn freeze(p: &Int8DocumentExtractionJobPlanner, input: &[u8]) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: p.execution_identity(), recipe: p.job_recipe(), limits: limits() },
        [JobInput { id: "item", original: input, normalized: input }], &mut Continue).unwrap()
}
#[test]
fn original_document_planning_uses_every_exact_child_and_one_whole_document_debit() {
    let p = factory(config()); p.check_job_limits(limits()).unwrap();
    let text = "éAéAéAéA";
    let plan = p.compiler.plan_document_with_control(text, &p.config.request, p.config.limits, &mut Continue).unwrap();
    let expected = plan.preflight_metadata();
    assert!(expected.chunk_count() > 1);
    assert_eq!(expected.source_span().byte_end, text.len());
    assert_eq!(expected.source_span().scalar_end, text.chars().count());
    let debit = p.document_work(); let actual = expected.planned_work();
    assert!(JobWork { model: actual, mask_node_visits: expected.reserved_mask_visits() }.fits(debit));
    assert!(debit.model.forward_positions > actual.forward_positions);
    for child in plan.execution_identities() {
        assert_eq!(child.template_digest, p.execution_identity().template_digest);
        assert_eq!(child.tokenizer_digest, p.execution_identity().tokenizer_digest);
        assert_eq!(child.schema_digest, Sha256Digest::of_bytes(p.config.request.schema.as_bytes()));
    }
    assert_ne!(p.execution_identity().tokenizer_digest, identity().tokenizer_digest);
}
#[test]
fn exact_schema_bytes_and_original_population_have_distinct_authenticated_bindings() {
    let mut c = config();
    let schema = " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678}\n";
    c.request.schema = schema.to_owned(); let a = factory(c.clone());
    assert_eq!(serde_json::to_value(a.job_recipe()).unwrap()["request"]["schema"].as_str(), Some(schema));
    c.request.schema.push(' '); let b = factory(c);
    let input = br#"{"id":"item","text":"Alice"}"#;
    let first = freeze(&a, input); first.binding.compare(&freeze(&a, input).binding).unwrap();
    assert_eq!(first.binding.compare(&freeze(&b, input).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(first.binding.compare(&freeze(&a, br#"{"id":"item","text":"Bob"}"#).binding),
        Err(JobError::Mismatch(MismatchField::Population)));
}
#[test]
fn every_mapping_and_verification_axis_is_in_the_frozen_recipe() {
    let baseline = canonjson::canonical_bytes(factory(config()).job_recipe()).unwrap();
    for axis in 0..25 {
        let mut c = config(); let m = &mut c.limits.mapping;
        match axis {
            0 => m.chunks.max_input_bytes += 1, 1 => m.chunks.max_chunk_bytes += 1,
            2 => m.chunks.max_chunk_tokens += 1, 3 => m.chunks.context_tokens += 1,
            4 => m.chunks.reserved_tokens += 1, 5 => m.chunks.max_chunks += 1,
            6 => m.chunks.max_tokenizer_calls += 1, 7 => m.reduction.map_batch_chunks += 1,
            8 => m.reduction.reduce_fan_in += 1, 9 => m.reduction.max_reduction_levels += 1,
            10 => m.reduction.max_task_calls += 1, 11 => m.reduction.max_value_bytes += 1,
            12 => m.reduction.max_live_value_bytes += 1, 13 => m.reduction.max_total_value_bytes += 1,
            14 => m.reduction.max_result_bytes += 1, 15 => m.max_model_work.forward_positions += 1,
            16 => m.max_model_work.projected_logits += 1, 17 => m.max_model_work.attention_pairs += 1,
            18 => m.max_model_work.projections.dot_products += 1,
            19 => m.max_model_work.projections.multiply_accumulates += 1,
            20 => m.mask_visits_per_chunk += 1, 21 => m.max_mask_visits += 1,
            22 => c.limits.verification.max_fields += 1, 23 => c.limits.verification.max_matches += 1,
            _ => c.limits.verification.max_scan_steps += 1,
        }
        assert_ne!(baseline, canonjson::canonical_bytes(factory(c).job_recipe()).unwrap(), "axis {axis}");
    }
}
#[test]
fn verification_lifetime_covers_failed_attempts_not_just_successful_items() {
    let p = factory(config()); let mut l = limits();
    p.check_job_limits(l).unwrap();
    // Two inputs but four attempts are already priced. A fifth attempt could
    // consume a new verification allowance after an interrupted acknowledgement.
    l.max_attempts = 5; assert!(p.check_job_limits(l).is_err());
    for axis in 0..3 {
        let mut c = config();
        match axis { 0 => c.max_verification.max_fields -= 1,
            1 => c.max_verification.max_matches -= 1, _ => c.max_verification.max_scan_steps -= 1 }
        let p = factory(c); assert!(p.check_job_limits(limits()).is_err());
    }
    let c = config();
    assert!(!verification_covers_attempts(c.limits.verification, c.max_verification, 0));
    assert!(!verification_covers_attempts(c.limits.verification,
        GroundingBudget { max_fields: usize::MAX, max_matches: usize::MAX, max_scan_steps: u64::MAX }, u64::MAX));
}
#[test]
fn every_durable_work_axis_and_the_complete_result_must_fit() {
    let p = factory(config());
    for axis in 0..7 {
        let mut l = limits(); l.max_work = p.document_work();
        match axis { 0 => l.max_work.model.forward_positions -= 1,
            1 => l.max_work.model.projected_logits -= 1, 2 => l.max_work.model.attention_pairs -= 1,
            3 => l.max_work.model.projections.dot_products -= 1,
            4 => l.max_work.model.projections.multiply_accumulates -= 1,
            5 => l.max_work.mask_node_visits -= 1, _ => l.max_result_bytes -= 1 }
        assert!(p.check_job_limits(l).is_err(), "axis {axis}");
    }
    let mut l = limits(); l.max_work.mask_node_visits += 1;
    assert!(p.check_job_limits(l).is_err());
}
#[test]
fn record_arguments_cannot_replace_the_fixed_schema_or_resume_policy() {
    for args in [r#"{"schema":"{}"}"#, r#"{"grounding":"structural"}"#, r#"{"budget":{}}"#,
        r#"{"max_chunks":1}"#, r#"{"max_attempts":999}"#] {
        assert!(serde_json::from_str::<BatchDocument<ExtractionDocumentArgs>>(
            &format!(r#"{{"id":"item","text":"private","task_args":{args}}}"#)).is_err());
    }
    assert!(serde_json::from_str::<BatchDocument<ExtractionDocumentArgs>>(
        r#"{"id":"item","text":"private","task_args":{}}"#).is_ok());
}
// Inspect only the runner's pre-storage contract. Neither prepare nor execute
// can fabricate a native result; both panic if a refactor crosses that boundary.
struct NoNative<'a>(&'a Int8DocumentExtractionJobPlanner);
impl BatchProcessor for NoNative<'_> {
    type Args = ExtractionDocumentArgs; type Prepared = (); type Output = ();
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<(), BatchItemFailure> { panic!("no preparation") }
    fn planned_work(&self, _: &()) -> BatchWork { panic!("no work") }
    fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<(), BatchItemFailure> { panic!("no native") }
}
impl DurableBatchProcessor for NoNative<'_> {
    type Recipe = Int8DocumentExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.0.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.0.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.0.check_job_limits(limits) }
    fn durable_work(&self, _: &()) -> Result<JobWork, BatchItemFailure> { panic!("no debit") }
    fn max_result_bytes(&self, _: &()) -> u64 { panic!("no result") }
}
#[test]
fn runner_checks_attempt_authority_before_population_freeze_or_storage() {
    let p = factory(config()); let mut l = limits(); l.max_attempts = 5;
    let error = super::super::super::freeze(&JobSecret::from_bytes([7; 32]), JobId([9; 16]), l,
        &[], &NoNative(&p), &mut Continue).err().unwrap();
    assert_eq!(error, JobRunError::Processor(BatchCode::InvalidLimits.into()));
}
#[test]
fn compiler_ceiling_and_cancellation_cannot_be_bypassed_by_document_mode() {
    let registry = pinned_controls::pinned().unwrap(); let mut c = config(); c.request.budget.max_input_tokens += 1;
    assert!(Int8DocumentExtractionJobPlanner::pinned(registry.template_controls(), 166101, identity(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), c).is_err());
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let p = factory(config());
    let error = p.compiler.plan_document_with_control("Alice", &p.config.request, p.config.limits, &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}
