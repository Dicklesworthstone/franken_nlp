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
fn compact_policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn encoding_limits() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }
}
fn compact() -> Int8DocumentExtractionJobPlanner {
    factory(config()).with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap()
}
#[test]
fn compact_resume_authenticates_every_encoding_axis_and_keeps_spent_work_out_of_the_recipe() {
    let p = compact(); let input = r#"{"id":"item","text":"éAéAéAéA"}"#.as_bytes();
    let frozen = freeze(&p, input);
    let plan = p.compiler.base_planner().plan_document_with_control("éAéAéAéA", &p.config.request, p.config.limits, &mut Continue).unwrap();
    assert!(plan.preflight_metadata().chunk_count() > 1);
    for child in plan.execution_identities() {
        assert_eq!(child.template_digest, p.execution_identity().template_digest);
        assert_eq!(child.schema_digest, Sha256Digest::of_bytes(p.config.request.schema.as_bytes()));
    }
    assert!(p.encoding_work().unwrap().unwrap().input_bytes() > 0);
    frozen.binding.compare(&freeze(&p, input).binding).unwrap();
    let resumed = compact(); resumed.check_job_limits(limits()).unwrap();
    frozen.binding.compare(&freeze(&resumed, input).binding).unwrap();
    assert_eq!(resumed.encoding_work().unwrap(), Some(SafeBpeWork::default()));
    let recipe = serde_json::to_value(p.job_recipe()).unwrap();
    assert_eq!(recipe["extraction"]["input_encoding"], serde_json::to_value(compact_policy()).unwrap());
    assert_eq!(recipe["extraction"]["encoding_work_limits"], serde_json::to_value(encoding_limits()).unwrap());
    for axis in 0..8 {
        let UntrustedEncoding::SafeBpeV1(mut call) = compact_policy() else { unreachable!() };
        let mut total = encoding_limits();
        match axis { 0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
            2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
            4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
            6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1 }
        let changed = factory(config()).with_document_encoding(UntrustedEncoding::SafeBpeV1(call), Some(total)).unwrap();
        assert!(frozen.binding.compare(&freeze(&changed, input).binding).is_err(), "axis {axis}");
    }
}
#[test]
fn compact_job_reconfiguration_preserves_spent_work_and_refuses_incomplete_or_changed_policy() {
    assert!(factory(config()).with_document_encoding(compact_policy(), None).is_err());
    assert!(factory(config()).with_document_encoding(UntrustedEncoding::ByteFallback, Some(encoding_limits())).is_err());
    for axis in 0..4 {
        let mut total = encoding_limits();
        match axis { 0 => total.max_input_bytes = 0, 1 => total.max_output_tokens = 0,
            2 => total.max_pair_checks = 0, _ => total.max_pair_bytes = 0 }
        assert!(factory(config()).with_document_encoding(compact_policy(), Some(total)).is_err());
    }
    let p = compact();
    let first = p.compiler.base_planner().plan_document_with_control("éAéAéAéA", &p.config.request, p.config.limits, &mut Continue).unwrap();
    let expected = first.preflight_metadata(); let ids: Vec<_> = first.execution_identities().cloned().collect();
    drop(first); let spent = p.encoding_work().unwrap().unwrap();
    let p = p.with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    assert_eq!(p.encoding_work().unwrap(), Some(spent));
    let rebuilt = p.compiler.base_planner().plan_document_with_control("éAéAéAéA", &p.config.request, p.config.limits, &mut Continue).unwrap();
    assert_eq!(rebuilt.preflight_metadata(), expected);
    assert_eq!(rebuilt.execution_identities().cloned().collect::<Vec<_>>(), ids);
    assert_eq!(p.encoding_work().unwrap(), Some(spent.checked_add(spent).unwrap()));
    drop(rebuilt);
    let mut changed = encoding_limits(); changed.max_input_bytes += 1;
    assert_eq!(p.with_document_encoding(compact_policy(), Some(changed)).err().unwrap().code, BatchCode::InvalidExecution);
    assert_eq!(compact().with_document_encoding(UntrustedEncoding::ByteFallback, None).err().unwrap().code,
        BatchCode::InvalidExecution);
}
#[test]
fn byte_job_defaults_preserve_the_original_recipe_identity_and_workspace() {
    let p = factory(config()); let recipe = canonjson::canonical_bytes(p.job_recipe()).unwrap();
    let identity = p.execution_identity().clone();
    assert_eq!(p.encoding_work_limits(), None); assert_eq!(p.encoding_work().unwrap(), None);
    let p = p.with_document_encoding(UntrustedEncoding::ByteFallback, None).unwrap();
    assert_eq!(p.document_encoding(), &UntrustedEncoding::ByteFallback);
    assert_eq!(p.execution_identity(), &identity); assert_eq!(p.encoding_workspace_bytes().unwrap(), 0);
    assert_eq!(canonjson::canonical_bytes(p.job_recipe()).unwrap(), recipe);
    let value = serde_json::to_value(p.job_recipe()).unwrap();
    assert!(value["extraction"].get("input_encoding").is_none());
    assert!(value["extraction"].get("encoding_work_limits").is_none());
    assert!(value.get("root_object_reconciliation").is_none());
    assert_eq!(value["execution"], INT8_EXTRACTION_MAP_EXECUTION);
    assert_eq!(value["extraction"]["prompt_version"], EXTRACTION_BATCH_PROMPT);
}
#[test]
fn compact_job_workspace_is_additive_to_the_complete_document_frontier() {
    let byte = factory(config()); let compact = compact();
    let UntrustedEncoding::SafeBpeV1(call) = compact_policy() else { unreachable!() };
    let extra = call.requirements().unwrap().workspace_bytes + 1024;
    assert_eq!(byte.document_work(), compact.document_work());
    assert_eq!(compact.encoding_workspace_bytes().unwrap(), extra);
    #[cfg(feature = "asupersync-runtime")]
    {
        use crate::hosted::corpus::jobs::extraction_document_job_workspace;
        let original = extraction_document_job_workspace(&byte, 4096).unwrap();
        assert_eq!(extraction_document_job_workspace(&compact, 4096).unwrap(), original + extra);
        assert!(extraction_document_job_workspace(&compact, u64::MAX).is_err());
    }
}
#[test]
fn a_changed_encoding_recipe_or_identity_refuses_before_storage() {
    for axis in 0..4 {
        let mut p = compact();
        match axis { 0 => p.recipe.extraction.input_encoding = None,
            1 => p.recipe.extraction.encoding_work_limits = None,
            2 => p.recipe.extraction.input_encoding = Some(UntrustedEncoding::ByteFallback),
            _ => p.identity.template_digest = Sha256Digest::of_bytes(b"foreign-compiler") }
        let error = super::super::super::freeze(&JobSecret::from_bytes([7; 32]), JobId([9; 16]), limits(),
            &[], &NoNative(&p), &mut Continue).err().unwrap();
        assert_eq!(error, JobRunError::Processor(BatchCode::Admission.into()), "axis {axis}");
    }
}
#[test]
fn original_document_planning_uses_every_exact_child_and_one_whole_document_debit() {
    let p = factory(config()); p.check_job_limits(limits()).unwrap();
    let text = "éAéAéAéA";
    let plan = p.compiler.base_planner().plan_document_with_control(text, &p.config.request, p.config.limits, &mut Continue).unwrap();
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
    let error = p.compiler.base_planner().plan_document_with_control("Alice", &p.config.request, p.config.limits, &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}

const OBJECT_SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"city":{"type":"string","enum":["Paris"]},"name":{"type":"string","enum":["Alice"]}},"required":["city","name"]}"#;
fn object_limits() -> RootObjectLimits {
    RootObjectLimits { max_properties: 8, max_origins: 32, max_input_bytes: 262144,
        max_json_bytes: 4096, max_nodes: 1024, max_result_bytes: 32768 }
}
fn object_config(schema: &str) -> ExtractionDocumentBatchConfig {
    let mut c = config(); c.request.schema = schema.to_owned();
    c.limits.mapping.reduction.map_batch_chunks = 2;
    c.limits.mapping.reduction.reduce_fan_in = 2;
    c
}
fn object_factory(schema: &str, reconciliation: RootObjectLimits) -> Int8DocumentExtractionJobPlanner {
    factory(object_config(schema)).with_root_object_reconciliation(reconciliation).unwrap()
}

#[test]
fn object_jobs_bind_the_original_schema_and_actual_partial_prompt_before_storage() {
    let original = format!("  {OBJECT_SCHEMA}\n");
    let p = object_factory(&original, object_limits());
    p.check_job_limits(limits()).unwrap();
    assert_eq!(p.request().schema, original);
    assert_eq!(p.root_object_reconciliation(), Some(object_limits()));
    assert_eq!(p.compiler_limits().max_schema_bytes, CompileLimits::default().max_schema_bytes);
    let recipe = serde_json::to_value(p.job_recipe()).unwrap();
    assert_eq!(recipe["request"]["schema"], original);
    assert_eq!(recipe["execution"], INT8_OBJECT_MAP_EXECUTION);
    assert_eq!(recipe["extraction"]["prompt_version"], ROOT_OBJECT_MAP_PROMPT);
    assert_eq!(recipe["root_object_reconciliation"]["policy"], ROOT_OBJECT_RECONCILIATION);
    assert_eq!(recipe["root_object_reconciliation"]["schema_derivation"], ROOT_OBJECT_DERIVATION);
    assert_eq!(recipe["root_object_reconciliation"]["prompt_version"], ROOT_OBJECT_MAP_PROMPT);
    assert_ne!(p.execution_identity().template_digest, factory(config()).execution_identity().template_digest);
    let source = "Alice Paris";
    let plan = p.compiler.plan(source, p.request(), p.document_limits(), &mut Continue).unwrap();
    let expected = plan.preflight_metadata();
    assert!(expected.chunk_count() > 1);
    assert_eq!(expected.source_span().byte_end, source.len());
    for child in plan.execution_identities() {
        assert_eq!(child.template_digest, p.execution_identity().template_digest);
        assert_eq!(child.tokenizer_digest, p.execution_identity().tokenizer_digest);
        assert_ne!(child.schema_digest, Sha256Digest::of_bytes(original.as_bytes()));
    }
    assert!(JobWork { model: expected.planned_work(), mask_node_visits: expected.reserved_mask_visits() }
        .fits(p.document_work()));
    assert_eq!(p.document_work(), factory(config()).document_work());
}

#[test]
fn original_required_sets_cannot_share_durable_authority_despite_identical_partial_maps() {
    let a = object_factory(OBJECT_SCHEMA, object_limits());
    let changed = OBJECT_SCHEMA.replace(r#""required":["city","name"]"#, r#""required":["name"]"#);
    assert_ne!(changed, OBJECT_SCHEMA);
    let b = object_factory(&changed, object_limits());
    let source = "Alice Paris";
    let first = a.compiler.plan(source, a.request(), a.document_limits(), &mut Continue).unwrap();
    let second = b.compiler.plan(source, b.request(), b.document_limits(), &mut Continue).unwrap();
    assert_eq!(first.execution_identities().cloned().collect::<Vec<_>>(),
        second.execution_identities().cloned().collect::<Vec<_>>());
    assert!(first.preflight_metadata() != second.preflight_metadata());
    let input = br#"{"id":"item","text":"Alice Paris"}"#;
    assert_eq!(freeze(&a, input).binding.compare(&freeze(&b, input).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    let spaced = object_factory(&format!("{OBJECT_SCHEMA}\n"), object_limits());
    assert_eq!(freeze(&a, input).binding.compare(&freeze(&spaced, input).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}

#[test]
fn every_root_object_limit_authenticates_resume_without_renewing_native_work() {
    let input = br#"{"id":"item","text":"Alice Paris"}"#;
    let p = object_factory(OBJECT_SCHEMA, object_limits());
    let frozen = freeze(&p, input);
    frozen.binding.compare(&freeze(&object_factory(OBJECT_SCHEMA, object_limits()), input).binding).unwrap();
    for axis in 0..6 {
        let mut changed = object_limits();
        match axis { 0 => changed.max_properties += 1, 1 => changed.max_origins += 1,
            2 => changed.max_input_bytes += 1, 3 => changed.max_json_bytes += 1,
            4 => changed.max_nodes += 1, _ => changed.max_result_bytes += 1 }
        let other = object_factory(OBJECT_SCHEMA, changed);
        assert_eq!(other.document_work(), p.document_work());
        assert_eq!(frozen.binding.compare(&freeze(&other, input).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
    }
    let mut too_many_attempts = limits(); too_many_attempts.max_attempts += 1;
    assert!(p.check_job_limits(too_many_attempts).is_err());
    let mut short_result = limits(); short_result.max_result_bytes -= 1;
    assert!(p.check_job_limits(short_result).is_err());
}

#[test]
fn object_recipe_drift_is_rejected_before_population_or_storage() {
    for axis in 0..8 {
        let mut p = object_factory(OBJECT_SCHEMA, object_limits());
        match axis {
            0 => p.recipe.root_object_reconciliation = None,
            1 => p.recipe.root_object_reconciliation.as_mut().unwrap().policy = "different-reducer",
            2 => p.recipe.root_object_reconciliation.as_mut().unwrap().schema_derivation = "different-derivation",
            3 => p.recipe.root_object_reconciliation.as_mut().unwrap().prompt_version = "different-prompt",
            4 => p.recipe.root_object_reconciliation.as_mut().unwrap().limits.max_properties += 1,
            5 => p.recipe.extraction.prompt_version = EXTRACTION_BATCH_PROMPT,
            6 => p.recipe.execution = INT8_EXTRACTION_MAP_EXECUTION,
            _ => p.identity.template_digest = Sha256Digest::of_bytes(b"other-template"),
        }
        let error = super::super::super::freeze(&JobSecret::from_bytes([7; 32]), JobId([9; 16]), limits(),
            &[], &NoNative(&p), &mut Continue).err().unwrap();
        assert_eq!(error, JobRunError::Processor(BatchCode::Admission.into()), "axis {axis}");
    }
}

#[test]
fn both_object_compact_setup_orders_rebuild_exactly_and_identical_settings_preserve_spent_work() {
    let configured = |encoding_first| {
        let p = factory(object_config(OBJECT_SCHEMA));
        if encoding_first {
            p.with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap()
                .with_root_object_reconciliation(object_limits()).unwrap()
        } else {
            p.with_root_object_reconciliation(object_limits()).unwrap()
                .with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap()
        }
    };
    let input = br#"{"id":"item","text":"Alice Paris"}"#;
    // The existing template wrappers bind their installation order. Resume
    // authenticates the actual execution identity as well as the same recipe.
    assert_eq!(freeze(&configured(false), input).binding.compare(&freeze(&configured(true), input).binding),
        Err(JobError::Mismatch(MismatchField::Execution)));
    for encoding_first in [false, true] {
        let p = configured(encoding_first); let frozen = freeze(&p, input);
        frozen.binding.compare(&freeze(&configured(encoding_first), input).binding).unwrap();
        let first = p.compiler.plan("Alice Paris", p.request(), p.document_limits(), &mut Continue).unwrap();
        let expected = first.preflight_metadata();
        let identities = first.execution_identities().cloned().collect::<Vec<_>>();
        drop(first);
        let spent = p.encoding_work().unwrap().unwrap(); assert!(spent.input_bytes() > 0);
        let p = p.with_root_object_reconciliation(object_limits()).unwrap()
            .with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
        assert_eq!(p.encoding_work().unwrap(), Some(spent));
        frozen.binding.compare(&freeze(&p, input).binding).unwrap();
        let rebuilt = p.compiler.plan("Alice Paris", p.request(), p.document_limits(), &mut Continue).unwrap();
        assert_eq!(rebuilt.preflight_metadata(), expected);
        assert_eq!(rebuilt.execution_identities().cloned().collect::<Vec<_>>(), identities);
        drop(rebuilt);
        assert_eq!(p.encoding_work().unwrap(), Some(spent.checked_add(spent).unwrap()));
        let mut changed = object_limits(); changed.max_origins += 1;
        assert!(p.with_root_object_reconciliation(changed).is_err());
    }
}

#[test]
fn object_reconstruction_exhausts_the_same_declared_compact_invocation_allowance() {
    let p = object_factory(OBJECT_SCHEMA, object_limits())
        .with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    drop(p.compiler.plan("Alice Paris", p.request(), p.document_limits(), &mut Continue).unwrap());
    let used = p.encoding_work().unwrap().unwrap();
    let once = SafeBpeWorkLimits { max_input_bytes: used.input_bytes() as u64,
        max_output_tokens: (used.output_tokens() as u64).max(1), max_pair_checks: used.pair_checks().max(1),
        max_pair_bytes: used.pair_bytes().max(1) };
    let p = object_factory(OBJECT_SCHEMA, object_limits()).with_document_encoding(compact_policy(), Some(once)).unwrap();
    drop(p.compiler.plan("Alice Paris", p.request(), p.document_limits(), &mut Continue).unwrap());
    assert_eq!(p.encoding_work().unwrap(), Some(used));
    assert!(p.compiler.plan("Alice Paris", p.request(), p.document_limits(), &mut Continue).is_err());
    assert_eq!(p.encoding_work().unwrap().unwrap().input_bytes(), used.input_bytes());
    // A separately authenticated resume gets this same bounded invocation
    // allowance, never a larger native/attempt/verification lifetime budget.
    let resumed = object_factory(OBJECT_SCHEMA, object_limits()).with_document_encoding(compact_policy(), Some(once)).unwrap();
    assert_eq!(p.document_work(), resumed.document_work());
    let input = br#"{"id":"item","text":"Alice Paris"}"#;
    freeze(&p, input).binding.compare(&freeze(&resumed, input).binding).unwrap();
}

#[test]
fn object_jobs_refuse_nonobjects_invalid_limits_and_changed_verification_authority() {
    for schema in [r#"{"type":"boolean"}"#,
        r#"{"type":"array","items":{"type":"boolean"},"maxItems":2}"#,
        r#"{"type":"object","properties":{},"additionalProperties":true}"#] {
        assert!(factory(object_config(schema)).with_root_object_reconciliation(object_limits()).is_err());
    }
    for axis in 0..6 {
        let mut invalid = object_limits();
        match axis { 0 => invalid.max_properties = 0, 1 => invalid.max_origins = 0,
            2 => invalid.max_input_bytes = 0, 3 => invalid.max_json_bytes = 0,
            4 => invalid.max_nodes = 0, _ => invalid.max_result_bytes = 0 }
        assert!(factory(object_config(OBJECT_SCHEMA)).with_root_object_reconciliation(invalid).is_err(), "axis {axis}");
    }
    for axis in 0..3 {
        let mut c = object_config(OBJECT_SCHEMA);
        match axis { 0 => c.max_verification.max_fields -= 1,
            1 => c.max_verification.max_matches -= 1, _ => c.max_verification.max_scan_steps -= 1 }
        let p = factory(c).with_root_object_reconciliation(object_limits()).unwrap();
        assert!(p.check_job_limits(limits()).is_err(), "axis {axis}");
    }
}

#[test]
fn object_job_preparation_retains_the_original_cancellation_without_spending_work() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    let p = object_factory(OBJECT_SCHEMA, object_limits())
        .with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        let error = p.compiler.plan("Alice Paris", p.request(), p.document_limits(), &mut Stop(cause)).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
        assert_eq!(object_failure(error), BatchFault::cancelled(cause));
        assert_eq!(p.encoding_work().unwrap(), Some(SafeBpeWork::default()));
    }
}
