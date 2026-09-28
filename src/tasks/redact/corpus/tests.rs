//! Model-free planning, receipt-corruption and transaction-lifetime tests.
//! Private synthetic completions test rejection only, never native/quality evidence.
use super::*;
use std::{cell::RefCell, rc::Rc};
use crate::{
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int8, decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits}, ner::{NerOptions, NER_TASK_VERSION},
        source_planning::{SourcePlanningLimits, quantized::long::Int8SourceMapLimits}},
    tokenizer::pinned_controls,
};
fn planner() -> SourceTaskPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    SourceTaskPlanner::pinned(controls, eos).unwrap()
}
fn config() -> LongRedactionBatchConfig {
    let p = planner(); let d = Sha256Digest::of_bytes(b"document-corpus-model-free-fixture");
    let one = constrained_int8::planned_work(128, 16).unwrap();
    let two = one.checked_add(one).unwrap(); let per = two.checked_add(two).unwrap();
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    LongRedactionBatchConfig { ner_identity: identity, detector: LongRedactionConfig {
        ner: NerOptions::default(), per_chunk: TaskBudget { max_input_tokens: 8192, max_output_tokens: 16,
            max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
        planning: SourcePlanningLimits::default(), mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
                context_tokens: 8192, reserved_tokens: 16, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits::default(), max_model_work: per,
            mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 4000 },
        max_result_bytes: 4 << 20 }, request: RedactionRequest::default(),
        max_model_work: per.checked_add(per).unwrap().checked_add(per).unwrap(), max_mask_visits: 12000 }
}
fn context(n: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: n, epoch, input_line: n, byte_offset: 0 }
}
#[derive(Default)]
struct Control { calls: usize, stop_at: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; self.stop_at.filter(|&n| self.calls >= n).map(|_| DecodeCancellationKind::Deadline)
    }
}
type Trace = Rc<RefCell<Vec<&'static str>>>;
struct Guard(Trace);
impl Drop for Guard { fn drop(&mut self) { self.0.borrow_mut().push("guard"); } }
#[derive(Serialize)]
struct Body { text: String, #[serde(skip)] trace: Trace }
impl Drop for Body { fn drop(&mut self) { self.trace.borrow_mut().push("body"); } }
struct Admission { trace: Trace, drift: bool }
impl Int8RedactionBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int8RedactionAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.trace.borrow_mut().push("admit");
        assert_eq!(request.mask_node_visits, 4000);
        assert_eq!(request.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
        let mut identity = request.identity.clone();
        if self.drift { identity.prompt_digest = Sha256Digest::of_bytes(b"wrong"); }
        Ok((identity, Guard(self.trace.clone())))
    }
}
fn admission() -> Admission { Admission { trace: Rc::default(), drift: false } }
#[test]
fn records_cannot_change_actions_types_chunks_verification_or_keys() {
    for args in [r#"{"verify":false}"#, r#"{"chunked":false}"#, r#"{"ner":{}}"#,
        r#"{"key":"secret"}"#, r#"{"namespace":"other"}"#, r#"{"actions":{}}"#] {
        assert!(serde_json::from_str::<BatchDocument<RedactionBatchArgs>>(
            &format!(r#"{{"id":"opaque","text":"private","task_args":{args}}}"#)).is_err());
    }
}
#[test]
fn source_admission_is_whole_document_not_one_chunk() {
    let mut c = config(); c.detector.planning.max_input_bytes = 6;
    check_source("Alice Alice Alice", &c).unwrap();
    assert!(check_source("", &c).is_err());
    c.request.rule_budget.max_input_bytes = 4;
    assert!(!check_source("Alice", &c).err().unwrap().stop);
}
#[test]
fn logit_ceiling_keeps_every_admissible_complete_projection() {
    let mut c = config(); let expected = c.item_model_work();
    c.detector.mapping.max_model_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert_eq!(c.item_model_work(), expected); check_limits(&c).unwrap();
    c.detector.mapping.max_model_work.projected_logits = NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert!(check_limits(&c).is_err());
}
#[test]
fn both_stages_and_complete_map_storage_must_fit_configuration() {
    let mut c = config(); check_configuration(&planner(), &c).unwrap();
    c.detector.mapping.max_mask_visits = 1000; assert!(check_limits(&c).is_err());
    c.request.verify = false; check_limits(&c).unwrap();
    c.detector.mapping.reduction.max_value_bytes = 1; assert!(check_limits(&c).is_err());
}
#[test]
fn all_native_axes_and_masks_are_atomic_corpus_reservations() {
    for axis in 0..6 {
        let mut c = config(); let p = c.item_model_work();
        match axis { 0 => c.max_model_work.forward_positions = p.forward_positions - 1,
            1 => c.max_model_work.projected_logits = p.projected_logits - 1,
            2 => c.max_model_work.attention_pairs = p.attention_pairs - 1,
            3 => c.max_model_work.projections.dot_products = p.projections.dot_products - 1,
            4 => c.max_model_work.projections.multiply_accumulates = p.projections.multiply_accumulates - 1,
            _ => c.max_mask_visits = 3999 }
        let mut ledger = Ledger::default(); assert!(ledger.begin(&c, context(1, 1)).is_err());
        assert_eq!(ledger.reserved, Int8Work::default()); assert_eq!(ledger.masks, 0); assert!(ledger.failed);
    }
}
#[test]
fn errors_and_flush_epochs_never_refund_original_or_verification_work() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission();
    for seq in 1..=3 {
        let error = execute_reserved::<_, _, (), _>(&mut ledger, &c, context(seq, seq), 4096,
            &mut a, &mut Control::default(), |_| (Err(Int8SourceMapError::WorkLimit.into()), true)).err().unwrap();
        assert!(!error.stop); assert_eq!(ledger.masks, seq * 4000);
    }
    assert_eq!(ledger.reserved, c.max_model_work);
    assert!(ledger.begin(&c, context(4, 4)).is_err());
}
#[test]
fn overflow_and_replayed_sequence_stop_without_partial_counter_changes() {
    let c = config();
    let mut ledger = Ledger::default(); ledger.masks = u64::MAX;
    assert!(ledger.begin(&c, context(1, 1)).is_err()); assert_eq!(ledger.reserved, Int8Work::default());
    let mut ledger = Ledger::default(); ledger.reserved.forward_positions = u64::MAX;
    let before = ledger.reserved; assert!(ledger.begin(&c, context(1, 1)).is_err());
    assert_eq!(ledger.reserved, before); assert_eq!(ledger.masks, 0);
    let mut ledger = Ledger::default(); ledger.begin(&c, context(3, 1)).unwrap(); ledger.finish(&Ok::<(), BatchItemFailure>(()));
    assert_eq!(ledger.begin(&c, context(3, 2)).err().unwrap().fault.code, BatchCode::InvalidExecution);
}
#[test]
fn admission_drift_prevents_native_execution_and_consumes_the_reservation() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission(); a.drift = true;
    let error = execute_reserved::<_, _, (), _>(&mut ledger, &c, context(1, 1), 4096,
        &mut a, &mut Control::default(), |_| panic!("must not execute")).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Admission); assert!(ledger.failed);
    assert_eq!(ledger.reserved, c.item_model_work()); assert_eq!(*a.trace.borrow(), ["admit", "guard"]);
}
#[test]
fn output_storage_outlives_execution_but_drops_before_admission_guard() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission(); let trace = a.trace.clone();
    let out = execute_reserved(&mut ledger, &c, context(1, 1), 4096, &mut a, &mut Control::default(), |_| {
        (Ok(Body { text: "safe".to_owned(), trace: trace.clone() }), true)
    }).unwrap();
    assert_eq!(*trace.borrow(), ["admit"]); drop(out);
    assert_eq!(*trace.borrow(), ["admit", "body", "guard"]); assert!(!ledger.failed);
}
#[test]
fn late_cancellation_and_dirty_native_state_never_publish_text() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission(); let trace = a.trace.clone();
    let error = execute_reserved(&mut ledger, &c, context(1, 1), 4096, &mut a,
        &mut Control { calls: 0, stop_at: Some(3) }, |_| {
            (Ok(Body { text: "must-not-escape".to_owned(), trace: trace.clone() }), true)
        }).err().unwrap();
    assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline)); assert!(ledger.failed);
    assert_eq!(*trace.borrow(), ["admit", "body", "guard"]);
    let mut ledger = Ledger::default();
    let error = execute_reserved::<_, _, (), _>(&mut ledger, &c, context(1, 1), 4096, &mut a,
        &mut Control::default(), |_| (Err(Int8SourceMapError::WorkLimit.into()), false)).err().unwrap();
    assert!(error.stop); assert!(ledger.failed);
}
fn receipt(c: &LongRedactionBatchConfig) -> LongRedactionRun {
    let source = "alpha beta";
    let result = super::super::redact_rules(source, &c.request, None).unwrap();
    let one = constrained_int8::planned_work(128, 16).unwrap(); let two = one.checked_add(one).unwrap();
    let stage = || LongRedactionStage { preflight: LongRedactionPreflight { source_bytes: source.len(), source_scalars: source.len(),
        chunks: 2, planned_model_work: two, reserved_mask_node_visits: 2000 }, model_work: two, mask_node_visit_charge: 4 };
    LongRedactionRun { schema_version: 1, execution: LONG_REDACTION_EXECUTION, numerics_profile: STRICT_INT8_PROFILE,
        detector_scope: "whole-source-rules-independent-ner-chunks-v1", result, original: stage(), verification: Some(stage()),
        reserved_model_work: c.item_model_work(), model_work: c.item_model_work(), reserved_mask_node_visits: 4000,
        mask_node_visit_charge: 8, verification_scan_steps: 0,
        warnings: ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established", "clean_scan_and_pseudonyms_are_not_anonymization"] }
}
#[test]
fn completion_reconciles_each_stage_original_geometry_and_all_counters() {
    let c = config(); let original = receipt(&c).original.preflight;
    validate_result(&receipt(&c), original, &c).unwrap();
    for axis in 0..10 {
        let mut out = receipt(&c);
        match axis { 0 => out.original.preflight.source_bytes += 1,
            1 => out.verification.as_mut().unwrap().preflight.source_bytes += 1,
            2 => out.verification.as_mut().unwrap().preflight.source_scalars += 1,
            3 => out.verification.as_mut().unwrap().preflight.chunks += 1,
            4 => out.reserved_model_work.projections.multiply_accumulates += 1,
            5 => out.model_work.attention_pairs += 1,
            6 => out.reserved_mask_node_visits += 1, 7 => out.mask_node_visit_charge += 1,
            8 => out.verification = None, _ => out.verification_scan_steps = u64::MAX }
        assert!(validate_result(&out, original, &c).is_err(), "{axis}");
    }
}
#[test]
fn residual_coordinates_are_dropped_and_transport_errors_are_content_free() {
    let error = failure(LongRedactionError::Redaction(Int8RedactionError::Residual(super::super::pipeline::LeakReport {
        schema_version: 1, residuals: Vec::new(), rules: super::super::detectors::RuleSet::default(), model_types: Default::default(),
    })));
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Execution);
    let cancelled = failure(LongRedactionError::Redaction(Int8RedactionError::Cancelled(DecodeCancellationKind::Deadline)));
    assert!(cancelled.stop); assert_eq!(cancelled.fault.cancellation, Some(DecodeCancellationKind::Deadline));
}
