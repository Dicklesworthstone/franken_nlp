//! Pinned planning, private recipe and owned-storage fixtures. These definitions
//! execute no native model and provide no inference or quality qualification.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION, canonjson,
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{constrained_int4, decode::DecodeCancellationKind,
        generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE},
    tasks::answer::ANSWER_TASK_VERSION,
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};
mod storage;

pub(super) const WIDTH: usize = 127;
const SOURCE: &str = "Alice and Bob met in 上海. The final é tail names Carol.";
const QUESTION: &str = "Who is named? PRIVATE_QUESTION";
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
}
pub(super) fn config(mode: usize) -> Int4QuestionDocumentBatchConfig {
    let mut item = constrained_int4::planned_work(WIDTH, 8128, 64).unwrap();
    for _ in 0..8 { item = item.checked_add(item).unwrap(); }
    let total = item.checked_add(item).unwrap();
    let verification = GroundingBudget::default();
    let lifetime = GroundingBudget { max_fields: verification.max_fields * 4,
        max_matches: verification.max_matches * 4, max_scan_steps: verification.max_scan_steps * 4 };
    Int4QuestionDocumentBatchConfig { options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65_536,
            max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes },
        planning: SourcePlanningLimits::default(), mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 12, max_chunk_tokens: 12,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2, ..ExecutionLimits::default() },
            max_model_work: item, mask_limits: MaskWorkLimits::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 256000, verification },
        verification, synthesis: (mode > 0).then_some(QuestionSynthesisLimits {
            max_evidence_passages: 32, max_evidence_bytes: 4096 }),
        hierarchy: (mode > 1).then_some(QuestionHierarchyLimits {
            max_levels: 3, max_passes: 4, ..QuestionHierarchyLimits::default() }),
        max_question_bytes: 1024, max_model_work: total.checked_add(total).unwrap(),
        max_mask_visits: 1_024_000, max_map_verification: lifetime, max_verification: lifetime }
}
pub(super) fn limits(config: Int4QuestionDocumentBatchConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 8 << 20, max_spool_bytes: 32 << 20,
        max_materialized_bytes: 32 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(config.max_model_work, config.max_mask_visits) }
}
fn source_planner() -> Arc<SourceTaskPlanner> {
    let controls = pinned_controls::pinned().unwrap();
    Arc::new(SourceTaskPlanner::pinned(controls.template_controls(),
        EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap()).unwrap())
}
fn identity(planner: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"question-document-storage-planning-fixture-not-model-authority");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH)).unwrap()),
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: ANSWER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn compiler(config: Int4QuestionDocumentBatchConfig) -> Int4QuestionDocumentBatchPlanner {
    let planner = source_planner();
    Int4QuestionDocumentBatchPlanner::for_planning_fixture(Arc::clone(&planner), identity(&planner), WIDTH, config).unwrap()
}
fn factory(mode: usize, chunks: bool, groups: Option<usize>) -> Int4QuestionDocumentJobPlanner {
    let config = config(mode); let mut planner = compiler(config);
    if chunks { planner = planner.with_chunk_cohorts().unwrap(); }
    if let Some(groups) = groups { planner = planner.with_hierarchy_cohorts(groups).unwrap(); }
    Int4QuestionDocumentJobPlanner::new(planner, limits(config)).unwrap()
}
fn document(source: &str, question: &str) -> BatchDocument<QuestionDocumentArgs> {
    BatchDocument { id: "opaque-user-id".to_owned(), text: source.to_owned(),
        task_args: Some(QuestionDocumentArgs { question: question.to_owned() }) }
}
fn frozen(planner: &Int4QuestionDocumentJobPlanner, bytes: &[u8]) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract {
        job_id: JobId([9; 16]), execution: planner.execution_identity(), recipe: planner.job_recipe(),
        limits: planner.job_limits() }, [JobInput { id: "document", original: bytes, normalized: bytes }],
        &mut Continue).unwrap()
}
fn record(source: &str, question: &str) -> Vec<u8> {
    canonjson::canonical_bytes(&serde_json::json!({ "id": "document", "text": source,
        "task_args": { "question": question } })).unwrap()
}
fn axis(work: &mut Int4Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn all_modes_reuse_exact_complete_document_work_and_private_ownership() {
    let plain = factory(0, false, None);
    let plain_prepared = plain.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
    assert!(plain_prepared.native_chunks() > 1);
    let pass = constrained_int4::planned_work(WIDTH, 8128, 64).unwrap();
    for (mode, chunks, groups) in [(0, false, None), (0, true, None), (1, false, None),
        (1, true, None), (2, false, None), (2, false, Some(2)), (2, true, Some(2))] {
        let planner = factory(mode, chunks, groups);
        let prepared = planner.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
        let work = planner.durable_work(&prepared).unwrap();
        assert_eq!(work, JobWork::from_int4(prepared.planned_work(), prepared.reserved_mask_visits()));
        assert_eq!(work.int4_model_work().projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
        let passes = planner.config().continuation_passes();
        let mut expected = plain_prepared.planned_work();
        for _ in 0..passes { expected = expected.checked_add(pass).unwrap(); }
        assert_eq!(prepared.planned_work(), expected);
        assert_eq!(prepared.reserved_mask_visits(), plain_prepared.reserved_mask_visits() + passes as u64 * 1000);
        assert_eq!(planner.max_output_tokens(), if mode == 2 { 4 * 64 } else { 0 });
        assert_eq!(planner.uses_native_cohorts(), chunks || groups.is_some());
        assert_eq!(planner.durable_work(&plain_prepared).unwrap_err().fault.code, BatchCode::Admission);
    }
}

#[test]
fn per_record_question_and_complete_source_are_authenticated_without_metadata_disclosure() {
    let planner = factory(2, true, Some(2)); let original = record(SOURCE, QUESTION);
    let manifest = frozen(&planner, &original);
    for changed in [record(SOURCE, "Who is named? CHANGED_PRIVATE_QUESTION"),
        record("Alice and Bob met in 上海. The final é tail names Dave.", QUESTION)] {
        assert_eq!(manifest.binding.compare(&frozen(&planner, &changed).binding),
            Err(JobError::Mismatch(MismatchField::Population)));
    }
    let metadata = String::from_utf8(canonjson::canonical_bytes(&(&manifest.binding, &manifest.items)).unwrap()).unwrap();
    for private in [SOURCE, QUESTION, "PRIVATE_QUESTION", "max_quote_scalars", "max_evidence_passages"] {
        assert!(!metadata.contains(private));
    }
    let recipe = String::from_utf8(canonjson::canonical_bytes(planner.job_recipe()).unwrap()).unwrap();
    assert!(!recipe.contains(SOURCE)); assert!(!recipe.contains(QUESTION));
    assert_eq!(serde_json::from_str::<serde_json::Value>(&recipe).unwrap()["source"]["defaults"], serde_json::Value::Null);
}

#[test]
fn fixed_policies_synthesis_hierarchy_and_schedules_bind_resume() {
    let original = factory(2, false, None); let input = record(SOURCE, QUESTION);
    let original_binding = frozen(&original, &input).binding;
    for field in 0..13 {
        let mut config = config(2);
        match field {
            0 => config.options.max_answer_scalars -= 1,
            1 => config.options.max_citations -= 1,
            2 => config.options.max_quote_scalars -= 1,
            3 => config.max_question_bytes -= 1,
            4 => config.mapping.chunks.max_chunk_bytes += 1,
            5 => config.mapping.reduction.reduce_fan_in += 1,
            6 => config.planning.max_input_bytes -= 1,
            7 => config.planning.compiler.max_schema_bytes -= 1,
            8 => config.synthesis.as_mut().unwrap().max_evidence_passages -= 1,
            9 => config.synthesis.as_mut().unwrap().max_evidence_bytes -= 1,
            10 => config.hierarchy.as_mut().unwrap().max_levels += 1,
            11 => config.hierarchy.as_mut().unwrap().max_passes += 1,
            _ => config.hierarchy.as_mut().unwrap().max_tokenizer_bytes -= 1,
        }
        let changed = Int4QuestionDocumentJobPlanner::new(compiler(config), limits(config)).unwrap();
        assert_eq!(original_binding.compare(&frozen(&changed, &input).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
    for changed in [factory(0, false, None), factory(1, false, None), factory(2, true, None),
        factory(2, false, Some(2)), factory(2, true, Some(2)), factory(2, true, Some(3))] {
        assert_eq!(original_binding.compare(&frozen(&changed, &input).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}

#[test]
fn all_native_and_mask_ceiling_changes_refuse_before_storage() {
    let config = config(2); let original = limits(config);
    check_limits(config, original).unwrap();
    for field in 0..6 {
        let mut changed = original;
        if field == 5 { changed.max_work.mask_node_visits -= 1; }
        else { let mut native = changed.max_work.int4_model_work(); *axis(&mut native, field) -= 1;
            changed.max_work = JobWork::from_int4(native, changed.max_work.mask_node_visits); }
        assert_eq!(check_limits(config, changed), Err(BatchCode::InvalidLimits.into()), "axis {field}");
    }
    let planner = Int4QuestionDocumentJobPlanner::new(compiler(config), original).unwrap();
    for field in 0..4 {
        let mut changed = original;
        match field { 0 => changed.max_attempts += 1, 1 => changed.max_items += 1,
            2 => changed.max_input_bytes_per_item += 1, _ => changed.max_result_bytes += 1 }
        assert!(planner.check_job_limits(changed).is_err());
    }
}

#[test]
fn both_verification_pools_cover_every_attempt_and_overflow_cannot_widen_authority() {
    let original = config(2); let limits = limits(original);
    for field in 0..6 {
        let mut changed = original;
        let pool = if field < 3 { &mut changed.max_map_verification } else { &mut changed.max_verification };
        match field % 3 { 0 => pool.max_fields -= 1, 1 => pool.max_matches -= 1, _ => pool.max_scan_steps -= 1 }
        assert_eq!(check_limits(changed, limits), Err(BatchCode::InvalidLimits.into()), "axis {field}");
    }
    let maximum = GroundingBudget { max_fields: usize::MAX, max_matches: usize::MAX, max_scan_steps: u64::MAX };
    for field in 0..3 {
        let mut per = GroundingBudget { max_fields: 1, max_matches: 1, max_scan_steps: 1 };
        match field { 0 => per.max_fields = usize::MAX, 1 => per.max_matches = usize::MAX, _ => per.max_scan_steps = u64::MAX }
        assert!(verification_lifetime(per, maximum, 2).is_err());
    }
    assert!(verification_lifetime(GroundingBudget::default(), maximum, 0).is_err());
}

#[test]
fn independent_byte_caps_fit_but_actual_pair_and_complete_encoded_record_still_bound_input() {
    let mut config = config(0); let mut job = limits(config);
    config.mapping.chunks.max_input_bytes = 5000; config.max_question_bytes = 5000;
    job.max_input_bytes_per_item = 8192;
    check_limits(config, job).unwrap();
    let planner = Int4QuestionDocumentJobPlanner::new(compiler(config), job).unwrap();
    assert!(planner.checked_document(document(&"x".repeat(5000), "Who?")).is_ok());
    assert!(planner.checked_document(document("Alice", &"q".repeat(5000))).is_ok());
    assert_eq!(planner.checked_document(document(&"x".repeat(5000), &"q".repeat(5000)))
        .err().unwrap().fault.code, BatchCode::DocumentLimit);
    let mut missing = document(SOURCE, QUESTION); missing.task_args = None;
    assert_eq!(planner.prepare_with_control(missing, &mut Continue).err().unwrap().fault.code, BatchCode::Planning);
    for field in 0..2 {
        let mut oversized = config;
        if field == 0 { oversized.mapping.chunks.max_input_bytes = job.max_input_bytes_per_item + 1; }
        else { oversized.max_question_bytes = job.max_input_bytes_per_item + 1; }
        assert!(check_limits(oversized, job).is_err());
    }
}

#[test]
fn changed_schedule_cannot_spend_old_preparation_and_cancellation_stays_typed() {
    let config = config(2); let compiler = compiler(config);
    let prior = compiler.prepare_with_control(SOURCE.to_owned(), QuestionDocumentArgs { question: QUESTION.to_owned() },
        &mut Continue).unwrap();
    let planner = Int4QuestionDocumentJobPlanner::new(compiler.with_hierarchy_cohorts(2).unwrap(), limits(config)).unwrap();
    assert_eq!(planner.durable_work(&prior).unwrap_err().fault.code, BatchCode::Admission);
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota] {
        let failure = planner.prepare_with_control(document(SOURCE, QUESTION), &mut Stop(cause)).err().unwrap();
        assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(cause));
    }
}
