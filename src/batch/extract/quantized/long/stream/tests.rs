//! Scripted lifecycle mechanics only; these are not native/model-quality receipts.
use super::*;
use std::{cell::Cell, rc::Rc};
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::runtime::JsonProgram,
    native_engine::{constrained::JsonDecodeOutput, decode::DecodeCancellationKind,
        portable_int8::ProjectionWork, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::ScoreSpace, mapreduce::{ChunkLimits, ExecutionLimits}},
    tokenizer::pinned_controls,
};
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"required":["quote"],"properties":{"quote":{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}}}"#;
#[derive(Default)]
struct Control { calls: usize, stop: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::Deadline)
    }
    fn prefill_checkpoint(&mut self, n: usize) -> Option<DecodeCancellationKind> { self.checkpoint(n) }
}
fn config() -> ExtractionDocumentBatchConfig {
    let max = Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
        projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } };
    ExtractionDocumentBatchConfig {
        request: ExtractionBatchArgs { schema: SCHEMA.to_owned(), grounding: ExtractionBatchGrounding::SourceMembership,
            budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
                max_grammar_states: 4096, max_kv_bytes: 1 << 31 } },
        limits: Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 6, max_chunk_tokens: 8192,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2, max_value_bytes: 4 << 20,
                ..ExecutionLimits::default() }, max_model_work: max, mask_limits: MaskWorkLimits::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 256_000 }, verification: GroundingBudget::default() },
        max_model_work: max, max_mask_visits: u64::MAX,
        max_verification: GroundingBudget { max_fields: usize::MAX, max_matches: usize::MAX, max_scan_steps: u64::MAX },
    }
}
fn planner(c: &ExtractionDocumentBatchConfig) -> Int8ExtractionBatchPlanner {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == IM_END).unwrap().id;
    let d = Sha256Digest::of_bytes(b"document-extraction-stream-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 1 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None };
    Int8ExtractionBatchPlanner::pinned(controls, eos, identity, c.request.budget,
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap()
}
#[derive(Default)]
struct Script { calls: usize, corrupt_at: Option<usize>, dirty: bool }
impl Driver for Script {
    fn capacity(&self) -> usize { 8192 }
    fn clean(&self) -> bool { !self.dirty }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction, _: &ExecutionIdentity,
        _: Int8JsonBudget, _: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        self.calls += 1;
        let source = prepared.source().text();
        let json = serde_json::to_string(&serde_json::json!({ "quote": source })).unwrap();
        let program = JsonProgram::compile_with_source(SCHEMA, source, CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        let options = prepared.extraction_plan().options();
        let token = (0..NANBEIGE_VOCAB_SIZE as u32).find(|id| !options.excluded_token_ids.contains(id) && *id != options.eos_token_id).unwrap();
        let work = constrained_int8::planned_work(prepared.extraction_plan().prompt_tokens(), 2).unwrap();
        let mut run = Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.to_owned(), model_work: work,
            result: ExtractResult { schema_version: 2, task_spec_version: "extract-v1".to_owned(), score_space: ScoreSpace::NotComputed,
                grounding: ExtractionGrounding::SourceMembership, source_fields: program.source_fields(&json).unwrap(),
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                    token_ids: vec![token, options.eos_token_id], json, forward_positions: work.forward_positions,
                    projected_logits: work.projected_logits, mask_node_visit_charge: 20 } } };
        if self.corrupt_at == Some(self.calls) { run.result.source_fields.clear(); }
        Ok(run)
    }
}
struct Guard(Rc<Cell<usize>>);
impl Drop for Guard { fn drop(&mut self) { self.0.set(self.0.get() - 1); } }
#[derive(Default)]
struct Admission { live: Rc<Cell<usize>>, calls: usize, drift: bool }
impl ExtractionDocumentBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: ExtractionDocumentAdmission<'_>) -> Result<(Vec<ExecutionIdentity>, Guard), BatchItemFailure> {
        self.calls += 1;
        assert!(!request.identities.is_empty());
        assert_eq!(request.mask_node_visits, request.identities.len() as u64 * 1000);
        assert_eq!(request.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
        let mut ids = request.identities.to_vec();
        if self.drift { ids.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"foreign-last-chunk"); }
        self.live.set(self.live.get() + 1);
        Ok((ids, Guard(Rc::clone(&self.live))))
    }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
fn prepare(p: &Int8ExtractionBatchPlanner, c: &ExtractionDocumentBatchConfig, d: &Script, l: &mut Ledger,
    source: &str) -> PreparedExtractionDocument {
    prepare_document(p, c, d, l, BatchDocument { id: "opaque".to_owned(), text: source.to_owned(), task_args: None },
        &mut Control::default()).unwrap()
}
#[test]
fn records_cannot_override_schema_grounding_geometry_or_budgets() {
    for args in [r#"{"schema":"{}"}"#, r#"{"grounding":"structural"}"#, r#"{"budget":{}}"#,
        r#"{"max_chunks":1}"#, r#"{"verification":false}"#] {
        assert!(serde_json::from_str::<BatchDocument<ExtractionDocumentArgs>>(
            &format!(r#"{{"id":"opaque","text":"private","task_args":{args}}}"#)).is_err());
    }
}
#[test]
fn two_whole_documents_reuse_the_driver_and_keep_output_guards_until_delivery() {
    let c = config(); c.validate().unwrap(); let p = planner(&c);
    let (mut d, mut a, mut l) = (Script::default(), Admission::default(), Ledger::default());
    let mut sum = Int8Work::default(); let mut chunks = 0;
    for (index, source) in ["éAéAéAéA", "Bob\r\n上海"].into_iter().enumerate() {
        let prepared = prepare(&p, &c, &d, &mut l, source);
        chunks += prepared.expected.chunk_count(); sum = add_work(sum, prepared.expected.planned_work()).unwrap();
        let out = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared,
            context(index as u64 + 1, index as u64 + 1), &mut Control::default()).unwrap();
        assert_eq!(a.live.get(), 1);
        let encoded = canonjson::canonical_string(&out).unwrap();
        assert!(encoded.contains("independent-chunk-json-no-global-object-merge-v1"));
        assert!(encoded.contains("original_fields"));
        drop(out); assert_eq!(a.live.get(), 0);
    }
    assert_eq!(d.calls, chunks); assert_eq!(a.calls, 2); assert_eq!(l.work, sum);
    assert_eq!(l.masks, chunks as u64 * 1000);
    assert_eq!(l.verification, add_verification(c.limits.verification, c.limits.verification).unwrap());
}
#[test]
fn every_native_mask_and_verification_axis_survives_an_epoch_change_without_refund() {
    let initial = config(); let p = planner(&initial); let source = "éAéAéAéA";
    let expected = prepare(&p, &initial, &Script::default(), &mut Ledger::default(), source).expected;
    for axis in 0..9 {
        let mut c = initial.clone(); c.limits.mapping.max_model_work = expected.planned_work();
        c.limits.mapping.max_mask_visits = expected.reserved_mask_visits();
        c.max_model_work = add_work(expected.planned_work(), expected.planned_work()).unwrap();
        c.max_mask_visits = 2 * expected.reserved_mask_visits();
        c.max_verification = add_verification(c.limits.verification, c.limits.verification).unwrap();
        match axis { 0 => c.max_model_work.forward_positions -= 1, 1 => c.max_model_work.projected_logits -= 1,
            2 => c.max_model_work.attention_pairs -= 1, 3 => c.max_model_work.projections.dot_products -= 1,
            4 => c.max_model_work.projections.multiply_accumulates -= 1, 5 => c.max_mask_visits -= 1,
            6 => c.max_verification.max_fields -= 1, 7 => c.max_verification.max_matches -= 1,
            _ => c.max_verification.max_scan_steps -= 1 }
        c.validate().unwrap();
        let (mut d, mut a, mut l) = (Script::default(), Admission::default(), Ledger::default());
        let first = prepare(&p, &c, &d, &mut l, source);
        drop(execute_document(&p, &c, &mut d, &mut a, &mut l, first, context(1, 1), &mut Control::default()).unwrap());
        let second = prepare(&p, &c, &d, &mut l, source);
        let error = execute_document(&p, &c, &mut d, &mut a, &mut l, second, context(2, 2), &mut Control::default()).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(d.calls, expected.chunk_count()); assert_eq!(a.calls, 1); assert!(l.failed);
        assert_eq!(l.work, expected.planned_work()); assert_eq!(l.masks, expected.reserved_mask_visits());
        assert_eq!(l.verification, c.limits.verification);
    }
}
#[test]
fn foreign_last_identity_is_refused_before_the_first_native_call() {
    let c = config(); let p = planner(&c);
    let (mut d, mut a, mut l) = (Script::default(), Admission { drift: true, ..Admission::default() }, Ledger::default());
    let prepared = prepare(&p, &c, &d, &mut l, "éAéAéAéA"); let expected = prepared.expected;
    assert!(execute_document(&p, &c, &mut d, &mut a, &mut l, prepared, context(1, 1), &mut Control::default()).is_err());
    assert_eq!(d.calls, 0); assert_eq!(a.live.get(), 0); assert!(l.failed);
    assert_eq!(l.work, expected.planned_work());
}
#[test]
fn recompiled_schema_or_source_drift_cannot_reuse_an_old_prepared_snapshot() {
    let c = config(); let p = planner(&c);
    for axis in 0..2 {
        let (mut d, mut a, mut l) = (Script::default(), Admission::default(), Ledger::default());
        let mut prepared = prepare(&p, &c, &d, &mut l, "Alice Bob");
        let mut changed = c.clone();
        if axis == 0 { changed.request.schema = changed.request.schema.replace("64", "63"); }
        else { prepared.source = "Carol Bob".to_owned(); }
        assert!(execute_document(&p, &changed, &mut d, &mut a, &mut l, prepared, context(1, 1), &mut Control::default()).is_err());
        assert_eq!(d.calls, 0); assert_eq!(a.calls, 0);
    }
}
#[test]
fn a_late_invalid_chunk_discards_the_document_and_permanently_closes_the_adapter() {
    let c = config(); let p = planner(&c);
    let (mut d, mut a, mut l) = (Script { corrupt_at: Some(2), ..Script::default() }, Admission::default(), Ledger::default());
    let prepared = prepare(&p, &c, &d, &mut l, "éAéAéAéA"); let expected = prepared.expected;
    let error = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared, context(1, 1), &mut Control::default()).err().unwrap();
    assert!(error.stop); assert_eq!(d.calls, 2); assert_eq!(a.live.get(), 0); assert!(l.failed);
    assert_eq!(l.work, expected.planned_work()); assert_eq!(l.masks, expected.reserved_mask_visits());
    assert!(prepare_document(&p, &c, &d, &mut l, BatchDocument { id: "retry".to_owned(), text: "Alice".to_owned(),
        task_args: None }, &mut Control::default()).is_err());
    assert_eq!(d.calls, 2);
}
#[test]
fn all_execution_checkpoints_preserve_cancellation_including_after_successful_maps() {
    let c = config(); let p = planner(&c); let source = "Alice Bob";
    let (mut d, mut a, mut l) = (Script::default(), Admission::default(), Ledger::default());
    let prepared = prepare(&p, &c, &d, &mut l, source); let mut counter = Control::default();
    drop(execute_document(&p, &c, &mut d, &mut a, &mut l, prepared, context(1, 1), &mut counter).unwrap());
    for stop in 1..=counter.calls {
        let (mut d, mut a, mut l) = (Script::default(), Admission::default(), Ledger::default());
        let prepared = prepare(&p, &c, &d, &mut l, source);
        let error = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared, context(1, 1),
            &mut Control { calls: 0, stop: Some(stop) }).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
        assert_eq!(a.live.get(), 0); assert!(l.failed);
    }
}
#[test]
fn sequence_replay_cannot_execute_again_after_a_successful_epoch() {
    let c = config(); let p = planner(&c);
    let (mut d, mut a, mut l) = (Script::default(), Admission::default(), Ledger::default());
    let prepared = prepare(&p, &c, &d, &mut l, "Alice");
    drop(execute_document(&p, &c, &mut d, &mut a, &mut l, prepared, context(1, 1), &mut Control::default()).unwrap());
    let prepared = prepare(&p, &c, &d, &mut l, "Alice");
    let error = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared, context(1, 2), &mut Control::default()).err().unwrap();
    assert!(error.stop); assert_eq!(d.calls, 1); assert_eq!(a.calls, 1);
}
