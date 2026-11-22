//! Model-free pinned planning and durable-contract regressions; not inference.
use super::*;
use crate::{canonjson, execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ner::NerOptions, keyphrases::KeyphraseOptions, summarize::SummaryOptions},
    tokenizer::pinned_controls};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(task: &SourceMapTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"source-document-job-fixture");
    let name = match task { SourceMapTask::Ner(_) => "ner-v1", SourceMapTask::Keyphrases(_) => "keyphrases-v1",
        SourceMapTask::Summarize(_) => "summarize-v1" };
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "fixture".into(), quant_recipe: "int8-fixture".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: name.into(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
fn maximum() -> JobWork {
    let mut model = Int8Work::default();
    model.forward_positions = u64::MAX; model.projected_logits = u64::MAX; model.attention_pairs = u64::MAX;
    model.projections.dot_products = u64::MAX; model.projections.multiply_accumulates = u64::MAX;
    JobWork { model, mask_node_visits: u64::MAX }
}
fn lifetime() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 4 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20, max_attempts: 4, max_work: maximum() }
}
fn mapping() -> Int8SourceMapLimits {
    Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8,
        max_chunk_tokens: 8, context_tokens: 8192, reserved_tokens: 1024,
        max_chunks: 256, max_tokenizer_calls: 1024 }, reduction: ExecutionLimits::default(),
        max_model_work: maximum().model, mask_limits: MaskWorkLimits::default(),
        mask_visits_per_chunk: 1000, max_mask_visits: 256_000 }
}
fn build(task: SourceMapTask, mapping: Int8SourceMapLimits) -> Int8SourceDocumentJobPlanner {
    let controls = pinned_controls::pinned().unwrap();
    Int8SourceDocumentJobPlanner::pinned(controls.template_controls(), 166101, identity(&task), task,
        budget(), SourcePlanningLimits::default(), mapping, lifetime(), &mut Continue).unwrap()
}
pub(crate) fn factory() -> Int8SourceDocumentJobPlanner { build(SourceMapTask::Ner(NerOptions::default()), mapping()) }
fn document(text: &str) -> BatchDocument<SourceDocumentArgs> {
    BatchDocument { id: "item".into(), text: text.into(), task_args: None }
}
fn freeze(p: &Int8SourceDocumentJobPlanner, text: &[u8]) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: p.execution_identity(), recipe: p.job_recipe(), limits: p.limits },
        [JobInput { id: "item", original: text, normalized: text }], &mut Continue).unwrap()
}
fn context(seq: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch: 1, input_line: seq, byte_offset: 0 }
}

const COMPACT_SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁";
fn compact_policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn compact_work() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }
}
fn compact_mapping() -> Int8SourceMapLimits {
    Int8SourceMapLimits { chunks: ChunkLimits { max_chunk_bytes: 4096, max_chunk_tokens: 4096,
        ..mapping().chunks }, ..mapping() }
}
fn compact_factory(total: SafeBpeWorkLimits) -> Int8SourceDocumentJobPlanner {
    build(SourceMapTask::Ner(NerOptions::default()), compact_mapping())
        .with_document_encoding(compact_policy(), Some(total)).unwrap()
}

#[test]
fn compact_jobs_keep_original_documents_and_exact_reconstruction_for_every_source_task() {
    for task in [SourceMapTask::Ner(NerOptions::default()), SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        SourceMapTask::Summarize(SummaryOptions::default())] {
        let byte = build(task, compact_mapping());
        let old = byte.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
        let compact = byte.with_document_encoding(compact_policy(), Some(compact_work())).unwrap();
        assert!(compact.owns(&old).is_err());
        let first = compact.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
        assert_eq!(first.source, COMPACT_SOURCE);
        assert_eq!(first.chunk_count(), old.chunk_count());
        assert!(first.work.model.forward_positions < old.work.model.forward_positions);
        assert_eq!(first.work.mask_node_visits, old.work.mask_node_visits);
        let used = compact.encoding_work().unwrap().unwrap();
        assert_eq!(used.input_bytes(), 2 * COMPACT_SOURCE.len());
        assert!(used.output_tokens() < used.input_bytes());
        let rebuilt = compact.plan(&first.source, &mut Continue).unwrap();
        assert!(snapshot_matches(&rebuilt, &first));
        assert_eq!(compact.encoding_work().unwrap().unwrap(), used.checked_add(used).unwrap());
        drop(rebuilt);
        let next = compact.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
        assert_eq!(first.identities, next.identities);
        assert_eq!(compact.encoding_work().unwrap().unwrap(), used.checked_add(used).unwrap().checked_add(used).unwrap());
        compact.check_job_limits(lifetime()).unwrap();
    }
}

#[test]
fn preparation_reconstruction_and_later_records_share_every_tokenizer_work_axis() {
    let probe = compact_factory(compact_work());
    let first = probe.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
    let plan = probe.plan(&first.source, &mut Continue).unwrap();
    assert!(snapshot_matches(&plan, &first));
    let used = probe.encoding_work().unwrap().unwrap();
    let exact = SafeBpeWorkLimits { max_input_bytes: used.input_bytes() as u64,
        max_output_tokens: used.output_tokens() as u64, max_pair_checks: used.pair_checks(), max_pair_bytes: used.pair_bytes() };
    let factory = compact_factory(exact);
    let first = factory.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
    let rebuilt = factory.plan(&first.source, &mut Continue).unwrap();
    assert!(snapshot_matches(&rebuilt, &first));
    drop(rebuilt);
    assert_eq!(factory.encoding_work().unwrap(), Some(used));
    assert!(factory.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).is_err());
    assert_eq!(factory.encoding_work().unwrap(), Some(used));
    for axis in 0..4 {
        let mut short = exact;
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        let limited = compact_factory(short);
        let first = limited.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
        assert!(limited.plan(&first.source, &mut Continue).is_err(), "axis {axis}");
        assert!(limited.encoding_work().unwrap().unwrap().input_bytes() >= 2 * COMPACT_SOURCE.len());
    }
}

#[test]
fn compact_configuration_cannot_reset_spent_work_or_accept_incomplete_admission() {
    assert!(factory().with_document_encoding(compact_policy(), None).is_err());
    assert!(factory().with_document_encoding(UntrustedEncoding::ByteFallback, Some(compact_work())).is_err());
    let p = compact_factory(compact_work());
    let first = p.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
    let spent = p.encoding_work().unwrap();
    let p = p.with_document_encoding(compact_policy(), Some(compact_work())).unwrap();
    p.owns(&first).unwrap(); assert_eq!(p.encoding_work().unwrap(), spent);
    let mut changed = compact_work(); changed.max_input_bytes += 1;
    assert_eq!(p.with_document_encoding(compact_policy(), Some(changed)).err().unwrap().code,
        BatchCode::InvalidExecution);
}

#[test]
fn compact_resume_authenticates_all_call_and_invocation_limits_without_persisting_tokenizer_counts() {
    let original = compact_factory(compact_work());
    let population = br#"{"id":"item","text":"private source"}"#;
    let frozen = freeze(&original, population);
    let prepared = original.prepare_with_control(document(COMPACT_SOURCE), &mut Continue).unwrap();
    let resumed = compact_factory(compact_work());
    frozen.binding.compare(&freeze(&resumed, population).binding).unwrap();
    assert_eq!(resumed.encoding_work().unwrap().unwrap(), crate::tokenizer::untrusted::SafeBpeWork::default());
    assert!(resumed.owns(&prepared).is_err());
    original.check_job_limits(lifetime()).unwrap(); resumed.check_job_limits(lifetime()).unwrap();
    for axis in 0..8 {
        let UntrustedEncoding::SafeBpeV1(mut call) = compact_policy() else { unreachable!() };
        let mut total = compact_work();
        match axis { 0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
            2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
            4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
            6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1 }
        let changed = build(SourceMapTask::Ner(NerOptions::default()), compact_mapping())
            .with_document_encoding(UntrustedEncoding::SafeBpeV1(call), Some(total)).unwrap();
        assert!(frozen.binding.compare(&freeze(&changed, population).binding).is_err(), "axis {axis}");
    }
    let byte = factory(); let original_recipe = canonjson::canonical_bytes(byte.job_recipe()).unwrap();
    let byte = byte.with_document_encoding(UntrustedEncoding::ByteFallback, None).unwrap();
    assert_eq!(canonjson::canonical_bytes(byte.job_recipe()).unwrap(), original_recipe);
}

#[test]
fn rejected_compact_partition_probes_spend_work_and_original_cancellation_survives() {
    let mut map = compact_mapping(); map.chunks.max_chunk_bytes = 64; map.chunks.max_chunk_tokens = 8;
    let p = build(SourceMapTask::Ner(NerOptions::default()), map)
        .with_document_encoding(compact_policy(), Some(compact_work())).unwrap();
    let text = format!("{COMPACT_SOURCE}\r\n").repeat(3);
    let prepared = p.prepare_with_control(document(&text), &mut Continue).unwrap();
    assert!(prepared.chunk_count() > 1); assert_eq!(prepared.source, text);
    assert!(p.encoding_work().unwrap().unwrap().input_bytes() > 2 * text.len());
    struct Stop { calls: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls >= 7).then_some(self.cause)
        }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let p = compact_factory(compact_work());
        let error = p.prepare_with_control(document(&text), &mut Stop { calls: 0, cause }).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.cancellation, Some(cause));
    }
}

#[test]
fn all_three_native_tasks_plan_complete_documents_and_bind_the_actual_assets() {
    for task in [SourceMapTask::Ner(NerOptions::default()), SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        SourceMapTask::Summarize(SummaryOptions::default())] {
        let original = identity(&task);
        let p = build(task, mapping());
        let prepared = p.prepare_with_control(document("Alice Bob Carol Dave"), &mut Continue).unwrap();
        assert!(prepared.chunk_count() > 1);
        assert_ne!(p.identity.tokenizer_digest, original.tokenizer_digest);
        assert_ne!(p.identity.template_digest, original.template_digest);
        for id in prepared.execution_identities() {
            assert_eq!(id.task_spec, p.identity.task_spec);
            assert_eq!(id.tokenizer_digest, p.identity.tokenizer_digest);
            assert_eq!(id.template_digest, p.identity.template_digest);
        }
        let again = p.plan(&prepared.source, &mut Continue).unwrap();
        assert!(snapshot_matches(&again, &prepared));
        assert_eq!(prepared.work.mask_node_visits, 1000 * prepared.chunk_count() as u64);
        assert!(prepared.work.fits(p.recipe.document_ceiling));
        assert_eq!(prepared.source, "Alice Bob Carol Dave");
    }
}
#[test]
fn prepared_documents_cannot_be_swapped_between_factories_or_rebound_to_other_text() {
    let a = factory(); let b = factory();
    let prepared = a.prepare_with_control(document("Alice Bob Carol Dave"), &mut Continue).unwrap();
    a.owns(&prepared).unwrap(); assert!(b.owns(&prepared).is_err());
    let changed = a.plan("Grace Bob Carol Dave", &mut Continue).unwrap();
    assert!(!snapshot_matches(&changed, &prepared));
    let mut changed = a.prepare_with_control(document("Alice Bob Carol Dave"), &mut Continue).unwrap();
    changed.identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"tampered last chunk");
    let actual = a.plan(&changed.source, &mut Continue).unwrap();
    assert!(!snapshot_matches(&actual, &changed));
}
#[test]
fn frozen_recipe_and_original_population_are_independently_authenticated() {
    let a = factory();
    let b = build(SourceMapTask::Ner(NerOptions::default()), Int8SourceMapLimits {
        chunks: ChunkLimits { max_chunk_bytes: 9, ..mapping().chunks }, ..mapping() });
    let text = br#"{"id":"item","text":"Alice"}"#;
    let first = freeze(&a, text);
    first.binding.compare(&freeze(&a, text).binding).unwrap();
    assert_eq!(first.binding.compare(&freeze(&b, text).binding), Err(crate::jobs::JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(first.binding.compare(&freeze(&a, br#"{"id":"item","text":"Grace"}"#).binding),
        Err(crate::jobs::JobError::Mismatch(MismatchField::Population)));
}
#[test]
fn partition_reduction_and_document_work_axes_are_all_bound() {
    let original = canonjson::canonical_bytes(factory().job_recipe()).unwrap();
    for axis in 0..21 {
        let mut m = mapping();
        match axis {
            0 => m.chunks.max_input_bytes += 1, 1 => m.chunks.max_chunk_bytes += 1,
            2 => m.chunks.max_chunk_tokens += 1, 3 => m.chunks.context_tokens -= 1,
            4 => m.chunks.reserved_tokens += 1, 5 => m.chunks.max_chunks -= 1,
            6 => m.chunks.max_tokenizer_calls += 1, 7 => m.reduction.map_batch_chunks += 1,
            8 => m.reduction.reduce_fan_in += 1, 9 => m.reduction.max_reduction_levels += 1,
            10 => m.reduction.max_task_calls += 1, 11 => m.reduction.max_value_bytes += 1,
            12 => m.reduction.max_live_value_bytes += 1, 13 => m.reduction.max_total_value_bytes -= 1,
            14 => m.reduction.max_result_bytes -= 1, 15 => m.max_model_work.forward_positions -= 1,
            16 => m.max_model_work.projected_logits -= 1, 17 => m.max_model_work.attention_pairs -= 1,
            18 => m.max_model_work.projections.dot_products -= 1,
            19 => m.max_model_work.projections.multiply_accumulates -= 1,
            _ => m.mask_visits_per_chunk += 1,
        }
        let changed = build(SourceMapTask::Ner(NerOptions::default()), m);
        assert_ne!(original, canonjson::canonical_bytes(changed.job_recipe()).unwrap(), "axis {axis}");
    }
}
#[test]
fn exact_lifetime_limits_cannot_be_renewed_by_a_new_invocation() {
    let p = factory(); p.check_job_limits(lifetime()).unwrap();
    for axis in 0..8 {
        let mut l = lifetime();
        match axis { 0 => l.max_attempts += 1, 1 => l.max_items += 1, 2 => l.max_snapshot_bytes += 1,
            3 => l.max_work.model.forward_positions -= 1, 4 => l.max_work.model.projected_logits -= 1,
            5 => l.max_work.model.attention_pairs -= 1, 6 => l.max_work.model.projections.dot_products -= 1,
            _ => l.max_work.model.projections.multiply_accumulates -= 1 }
        assert!(p.check_job_limits(l).is_err());
    }
    let mut l = lifetime(); l.max_work.mask_node_visits -= 1; assert!(p.check_job_limits(l).is_err());
}
#[test]
fn records_cannot_inject_a_schema_task_budget_or_chunk_policy() {
    for text in [r#"{}"#, r#" { } "#] { serde_json::from_str::<SourceDocumentArgs>(text).unwrap(); }
    for text in [r#"{"schema":"{}"}"#, r#"{"task":"answer"}"#, r#"{"max_chunks":1}"#,
        r#"{"budget":{}}"#, r#"{"options":{}}"#] {
        assert!(serde_json::from_str::<SourceDocumentArgs>(text).is_err());
    }
    let p = factory();
    assert!(p.prepare_with_control(document(""), &mut Continue).is_err());
    assert!(p.prepare_with_control(document(&"x".repeat(4097)), &mut Continue).is_err());
}
#[test]
fn factory_and_reconstruction_observe_the_same_cancellation_kind() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        Some(DecodeCancellationKind::Deadline)
    } }
    let controls = pinned_controls::pinned().unwrap(); let task = SourceMapTask::Ner(NerOptions::default());
    let error = Int8SourceDocumentJobPlanner::pinned(controls.template_controls(), 166101, identity(&task), task,
        budget(), SourcePlanningLimits::default(), mapping(), lifetime(), &mut Stop).err().unwrap();
    assert_eq!(error.cancellation, Some(DecodeCancellationKind::Deadline));
    let p = factory();
    let error = p.prepare_with_control(document("Alice Bob"), &mut Stop).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
}
#[test]
fn malformed_mapping_and_every_unfunded_native_axis_fail_before_planning() {
    for axis in 0..14 {
        let mut m = mapping();
        match axis { 0 => m.chunks.max_input_bytes = 0, 1 => m.chunks.max_chunks = 257,
            2 => m.reduction.map_batch_chunks = 0, 3 => m.reduction.reduce_fan_in = 1,
            4 => m.reduction.max_value_bytes = 0, 5 => m.reduction.max_result_bytes = 64 * 1024 * 1024 + 1,
            6 => m.mask_visits_per_chunk = 0, 7 => m.max_mask_visits = 1,
            8 => m.max_model_work.forward_positions = 0, 9 => m.max_model_work.projected_logits = 0,
            10 => m.max_model_work.attention_pairs = 0, 11 => m.max_model_work.projections.dot_products = 0,
            12 => m.max_model_work.projections.multiply_accumulates = 0, _ => m.mask_limits.checkpoint_interval_nodes = 0 }
        assert!(validate_mapping(m, budget(), SourcePlanningLimits::default(), lifetime()).is_err(), "axis {axis}");
    }
}
#[test]
fn ledger_updates_all_work_axes_atomically_and_never_refunds_a_failed_attempt() {
    let p = factory(); let prepared = p.prepare_with_control(document("Alice Bob Carol Dave"), &mut Continue).unwrap();
    let work = prepared.work; let mut l = lifetime(); l.max_work = work;
    let mut ledger = Ledger::default(); ledger.begin(work, context(1), l).unwrap();
    assert_eq!(ledger.work, work); assert_eq!(ledger.attempts, 1);
    assert!(ledger.ready().is_err()); // Failure/unwind before completion is terminal.
    ledger.failed = false; // Simulates only the private successful-completion branch.
    assert!(ledger.begin(work, context(2), l).is_err());
    assert_eq!(ledger.work, work); assert_eq!(ledger.attempts, 1); assert_eq!(ledger.sequence, 1);
    for axis in 0..6 {
        let mut limit = lifetime(); limit.max_work = work;
        match axis { 0 => limit.max_work.model.forward_positions -= 1, 1 => limit.max_work.model.projected_logits -= 1,
            2 => limit.max_work.model.attention_pairs -= 1, 3 => limit.max_work.model.projections.dot_products -= 1,
            4 => limit.max_work.model.projections.multiply_accumulates -= 1, _ => limit.max_work.mask_node_visits -= 1 }
        let mut ledger = Ledger::default(); assert!(ledger.begin(work, context(1), limit).is_err());
        assert_eq!(ledger.work, JobWork::default()); assert_eq!(ledger.attempts, 0);
    }
}
#[test]
fn sequence_attempt_and_capacity_contracts_are_not_just_prompt_checks() {
    let mut ledger = Ledger::default(); let mut l = lifetime(); l.max_attempts = 1;
    ledger.begin(JobWork::default(), context(4), l).unwrap(); ledger.failed = false;
    assert!(ledger.begin(JobWork::default(), context(5), l).is_err());
    let mut ledger = Ledger::default(); ledger.begin(JobWork::default(), context(4), lifetime()).unwrap(); ledger.failed = false;
    assert!(ledger.begin(JobWork::default(), context(4), lifetime()).is_err());
    assert!(capacity_bytes(true, 8192, 8192, budget().max_kv_bytes).is_ok());
    assert!(capacity_bytes(false, 8192, 8192, budget().max_kv_bytes).is_err());
    assert!(capacity_bytes(true, 8191, 8192, budget().max_kv_bytes).is_err());
    assert!(capacity_bytes(true, 8192, 8192, KV_BYTES_PER_TOKEN as u64).is_err());
    assert!(capacity_bytes(true, usize::MAX, 8192, u64::MAX).is_err());
}
#[test]
fn real_processor_and_owned_factory_cross_the_host_boundary_without_a_fake_native() {
    struct Refuse;
    impl SourceDocumentBatchAdmission for Refuse {
        type Guard = ();
        fn admit(&mut self, _: SourceDocumentAdmission<'_>) -> Result<(Vec<ExecutionIdentity>, ()), BatchItemFailure> {
            Err(BatchItemFailure::fatal(BatchCode::Admission))
        }
    }
    fn durable<P: DurableBatchProcessor>() {}
    fn send<T: Send + 'static>() {}
    durable::<Int8SourceDocumentJobProcessor<'static, 'static, 'static, 'static, Refuse>>();
    send::<Int8SourceDocumentJobPlanner>(); send::<PreparedSourceDocument>();
}
