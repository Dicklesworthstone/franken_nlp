//! Pinned plan and keyed manifest contracts. No native answer or successful
//! inference is substituted by synthetic data in these regression definitions.
use super::*;
use crate::{canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField,
        question_document::quantized::tests::{config, limits}},
    native_engine::{artifact_bridge::ArtifactIdentity, decode::DecodeCancellationKind,
        lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::STRICT_INT8_EXECUTION},
    tasks::answer::ANSWER_TASK_VERSION,
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls, untrusted::SafeBpeLimits}};

const SOURCE: &str = "Alice and Bob met in 上海. The final é tail names Carol.";
const QUESTION: &str = "Who is named? PRIVATE_QUESTION";
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) } }
fn compiler(config: Int8QuestionDocumentBatchConfig) -> Int8QuestionDocumentBatchPlanner {
    let controls = pinned_controls::pinned().unwrap();
    let planner = Arc::new(SourceTaskPlanner::pinned(controls.template_controls(),
        EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap()).unwrap());
    let d = Sha256Digest::of_bytes(b"int8-question-job-fixture-not-model-authority");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: ANSWER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let model = ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: identity.source_revision.clone(),
        recipe_id: identity.quant_recipe.clone(), source_root_sha256: d.to_hex(), logical_model_sha256: d.to_hex() };
    Int8QuestionDocumentBatchPlanner::for_model(&model, planner, identity, config).unwrap()
}
fn factory() -> Int8QuestionDocumentJobPlanner {
    let config = config(); Int8QuestionDocumentJobPlanner::new(compiler(config), limits(config)).unwrap()
}
fn document(source: &str, question: &str) -> BatchDocument<QuestionDocumentArgs> {
    BatchDocument { id: "opaque".to_owned(), text: source.to_owned(), task_args: Some(QuestionDocumentArgs { question: question.to_owned() }) }
}
fn record(source: &str, question: &str) -> Vec<u8> {
    canonjson::canonical_bytes(&serde_json::json!({"id":"document","text":source,"task_args":{"question":question}})).unwrap()
}
fn frozen(planner: &Int8QuestionDocumentJobPlanner, input: &[u8]) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: planner.execution_identity(), recipe: planner.job_recipe(), limits: planner.job_limits() },
        [JobInput { id: "document", original: input, normalized: input }], &mut Continue).unwrap()
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn durable_adapter_quotes_exact_complete_plan_and_refuses_foreign_preparation() {
    let planner = factory();
    let prepared = planner.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
    assert!(prepared.native_chunks() > 1);
    assert_eq!(planner.durable_work(&prepared).unwrap(), JobWork {
        model: prepared.planned_work(), mask_node_visits: prepared.reserved_mask_visits() });
    assert_eq!(prepared.planned_work().projected_logits,
        prepared.native_chunks() as u64 * u64::from(planner.task_budget().max_output_tokens) * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(planner.max_output_tokens(), 0);
    assert_eq!(factory().durable_work(&prepared).unwrap_err().fault.code, BatchCode::Admission);
    let mut changed = planner.job_limits(); changed.max_attempts += 1;
    assert!(planner.check_job_limits(changed).is_err());
}

#[test]
fn exact_source_and_question_are_authenticated_without_metadata_disclosure() {
    let planner = factory(); let input = record(SOURCE, QUESTION); let manifest = frozen(&planner, &input);
    for changed in [record(SOURCE, "Who is named? CHANGED_PRIVATE_QUESTION"),
        record("Alice and Bob met in 上海. The final é tail names Dave.", QUESTION)] {
        assert_eq!(manifest.binding.compare(&frozen(&planner, &changed).binding), Err(JobError::Mismatch(MismatchField::Population)));
    }
    let metadata = String::from_utf8(canonjson::canonical_bytes(&(&manifest.binding, &manifest.items)).unwrap()).unwrap();
    assert!(!metadata.contains(SOURCE)); assert!(!metadata.contains("PRIVATE_QUESTION"));
    let recipe = String::from_utf8(canonjson::canonical_bytes(planner.job_recipe()).unwrap()).unwrap();
    assert!(!recipe.contains(SOURCE)); assert!(!recipe.contains(QUESTION));
    assert_eq!(serde_json::from_str::<serde_json::Value>(&recipe).unwrap()["source"]["defaults"], serde_json::Value::Null);
}

#[test]
fn every_fixed_answer_chunk_and_verification_policy_is_bound_into_resume() {
    let input = record(SOURCE, QUESTION); let original = frozen(&factory(), &input).binding;
    for field in 0..9 {
        let mut config = config();
        match field { 0 => config.options.max_answer_scalars -= 1, 1 => config.options.max_citations -= 1,
            2 => config.options.max_quote_scalars -= 1, 3 => config.max_question_bytes -= 1,
            4 => config.mapping.chunks.max_chunk_bytes += 1, 5 => config.mapping.reduction.reduce_fan_in += 1,
            6 => config.planning.compiler.max_schema_bytes -= 1, 7 => config.verification.max_scan_steps -= 1,
            _ => config.mapping.mask_limits.checkpoint_interval_nodes += 1 }
        let changed = Int8QuestionDocumentJobPlanner::new(compiler(config), limits(config)).unwrap();
        assert_eq!(original.compare(&frozen(&changed, &input).binding), Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
}

#[test]
fn prompt_schedule_is_bound_and_cannot_spend_an_old_prepared_owner() {
    let compiler = compiler(config());
    let old = compiler.prepare_with_control(SOURCE.to_owned(), QuestionDocumentArgs { question: QUESTION.to_owned() }, &mut Continue).unwrap();
    let planner = Int8QuestionDocumentJobPlanner::new(compiler, limits(config())).unwrap()
        .with_layer_major_prefill(prefill(4)).unwrap();
    assert_eq!(planner.durable_work(&old).unwrap_err().fault.code, BatchCode::Admission);
    let current = planner.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
    planner.durable_work(&current).unwrap();
    assert_eq!(planner.prefill_limits().unwrap().max_batch_rows, 4);
    let input = record(SOURCE, QUESTION);
    assert_eq!(frozen(&factory(), &input).binding.compare(&frozen(&planner, &input).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    let other = factory().with_layer_major_prefill(prefill(2)).unwrap();
    assert_eq!(frozen(&other, &input).binding.compare(&frozen(&planner, &input).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
}

#[test]
fn complete_pair_input_and_original_cancellation_are_checked_before_preparation() {
    let mut config = config(); config.mapping.chunks.max_input_bytes = 5000; config.max_question_bytes = 5000;
    let planner = Int8QuestionDocumentJobPlanner::new(compiler(config), limits(config)).unwrap();
    assert!(planner.checked_document(document(&"x".repeat(5000), "Who?")).is_ok());
    assert!(planner.checked_document(document("Alice", &"q".repeat(5000))).is_ok());
    assert_eq!(planner.checked_document(document(&"x".repeat(5000), &"q".repeat(5000))).err().unwrap().fault.code, BatchCode::DocumentLimit);
    let mut missing = document(SOURCE, QUESTION); missing.task_args = None;
    assert_eq!(planner.prepare_with_control(missing, &mut Continue).err().unwrap().fault.code, BatchCode::Planning);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = planner.prepare_with_control(document(SOURCE, QUESTION), &mut Stop(cause)).err().unwrap();
        assert_eq!(error.fault, BatchFault::cancelled(cause)); assert!(error.stop);
    }
}

#[test]
fn compact_policy_and_same_pair_reconstruction_allowance_bind_the_recipe() {
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
    let total = SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 };
    let compact = factory().with_map_document_encoding(policy.clone(), total).unwrap();
    let input = record(SOURCE, QUESTION);
    // Encoding and its complete work policy bind the template identity before
    // the independent recipe commitment is compared by the durable runner.
    assert_eq!(frozen(&factory(), &input).binding.compare(&frozen(&compact, &input).binding), Err(JobError::Mismatch(MismatchField::Execution)));
    assert_ne!(canonjson::canonical_bytes(factory().job_recipe()).unwrap(),
        canonjson::canonical_bytes(compact.job_recipe()).unwrap());
    let first = compact.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
    let second = compact.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
    assert_eq!(first.map_document_encoding_work().unwrap(), second.map_document_encoding_work().unwrap());
    let mut changed = total; changed.max_pair_bytes -= 1;
    let changed = factory().with_map_document_encoding(policy, changed).unwrap();
    assert_eq!(frozen(&changed, &input).binding.compare(&frozen(&compact, &input).binding), Err(JobError::Mismatch(MismatchField::Execution)));
    assert_ne!(canonjson::canonical_bytes(changed.job_recipe()).unwrap(),
        canonjson::canonical_bytes(compact.job_recipe()).unwrap());
    let recipe = serde_json::to_value(compact.job_recipe()).unwrap();
    assert_eq!(recipe["input_encoding"]["ownership"], "one-original-source-question-prepare-and-execute-reconstruction-v1");
    assert!(recipe["input_encoding"]["accounting"].as_str().unwrap().contains("not-journal-debits"));
}
