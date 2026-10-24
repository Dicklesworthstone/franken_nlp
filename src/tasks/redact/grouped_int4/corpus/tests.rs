//! Synthetic transaction/receipt fixtures, never native inference or recall
//! evidence. They exercise the real resident ledger and shared redaction edits.
use super::*;
use std::{cell::RefCell, rc::Rc};
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int4, decode::DecodeCancellationKind,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits, TaskStage},
        ner::{EntityType, NerOptions, NER_TASK_VERSION},
        redact::{actions, pipeline, PiiKind, pseudonym::PseudonymKey, union::DetectedDocument},
        source_planning::{SourcePlanningLimits, grouped_int4::long::Int4SourceMapLimits}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};
const WIDTH: usize = 127;
fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let registry = pinned_controls::pinned().unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn config() -> Int4DocumentRedactionBatchConfig {
    let p = planner(); let d = Sha256Digest::of_bytes(b"int4-corpus-private-model-free-fixture");
    let one = constrained_int4::planned_work(WIDTH, 128, 16).unwrap();
    let two = one.checked_add(one).unwrap(); let per = two.checked_add(two).unwrap();
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH,
        )).unwrap()),
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: NER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    Int4DocumentRedactionBatchConfig { ner_identity: identity, detector: Int4LongRedactionConfig {
        ner: NerOptions { types: vec![EntityType::Person], ..NerOptions::default() },
        per_chunk: TaskBudget { max_input_tokens: 8192, max_output_tokens: 16,
            max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
        planning: SourcePlanningLimits::default(), mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
                context_tokens: 8192, reserved_tokens: 16, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits::default(), max_model_work: per,
            mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 4000,
            verification: GroundingBudget { max_fields: 128, max_matches: 128, max_scan_steps: 8192 } },
        max_result_bytes: 4 << 20 }, request: RedactionRequest::default(),
        max_model_work: per.checked_add(per).unwrap().checked_add(per).unwrap(), max_mask_visits: 12000 }
}
fn context(n: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: n, epoch, input_line: n * 2, byte_offset: n * 128 }
}
fn axis(work: &mut Int4Work, n: usize) -> &mut u64 {
    match n { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits, 2 => &mut work.attention_pairs,
        3 => &mut work.projections.dot_products, _ => &mut work.projections.multiply_accumulates }
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
struct Admission { trace: Trace, expected: Int4Work, drift: Option<usize>, refuse: bool }
impl Int4RedactionBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int4RedactionAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.trace.borrow_mut().push("admit");
        assert_eq!(request.model_work, self.expected);
        assert_eq!(request.mask_node_visits, 4000);
        assert_eq!(request.kv_reservation_bytes, 8192);
        assert_eq!(request.max_result_bytes, 4 << 20);
        assert!(request.mask_limits.max_trie_node_visits > 0);
        assert!(request.mask_limits.checkpoint_interval_nodes > 0);
        assert_eq!(request.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
        if self.refuse { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        let mut identity = request.identity.clone();
        if let Some(n) = self.drift {
            match n { 0 => identity.prompt_digest = Sha256Digest::of_bytes(b"wrong"),
                1 => identity.logical_model_digest = Sha256Digest::of_bytes(b"wrong"),
                2 => identity.packing_set_digest = Sha256Digest::of_bytes(b"wrong"),
                3 => identity.tokenizer_digest = Sha256Digest::of_bytes(b"wrong"),
                4 => identity.template_digest = Sha256Digest::of_bytes(b"wrong"),
                5 => identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
                6 => identity.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=32"),
                _ => identity.task_spec = "summarize-v1".to_owned() }
        }
        Ok((identity, Guard(self.trace.clone())))
    }
}
fn admission(c: &Int4DocumentRedactionBatchConfig) -> Admission {
    Admission { trace: Rc::default(), expected: c.item_model_work(), drift: None, refuse: false }
}
fn source_error(error: Int4SourceError) -> Int4LongRedactionError {
    Int4LongRedactionError::Map(Int4SourceMapError::Source(error))
}
fn planning_error(error: SourcePlanningError) -> Int4LongRedactionError {
    source_error(Int4SourceError::Planning(error))
}
fn json_error(error: JsonDecodeError) -> Int4LongRedactionError {
    source_error(Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Decode(error))))
}
#[test]
fn records_cannot_replace_the_resident_detector_or_secret_scope() {
    for args in [r#"{"verify":false}"#, r#"{"chunked":false}"#, r#"{"ner":{"types":["money"]}}"#,
        r#"{"key":"private"}"#, r#"{"namespace":"different"}"#, r#"{"actions":{}}"#,
        r#"{"max_model_work":{}}"#, r#"{"group_width":32}"#] {
        let record = format!(r#"{{"id":"opaque","text":"private","task_args":{args}}}"#);
        assert!(serde_json::from_str::<BatchDocument<RedactionBatchArgs>>(&record).is_err());
    }
    assert!(serde_json::from_str::<BatchDocument<RedactionBatchArgs>>(
        r#"{"id":"opaque","text":"private","task_args":{}}"#).is_ok());
}
#[test]
fn original_source_admission_uses_complete_document_bounds() {
    let mut c = config(); c.detector.planning.max_input_bytes = 6;
    check_source("Alice Alice Alice", &c).unwrap();
    for source in ["", &"x".repeat(4097)] {
        let error = check_source(source, &c).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::DocumentLimit); assert!(!error.stop);
    }
    c.request.rule_budget.max_input_bytes = 4;
    assert!(!check_source("Alice", &c).unwrap_err().stop);
}
#[test]
fn native_projection_rounding_preserves_every_executable_row() {
    let mut c = config(); let expected = c.item_model_work();
    c.detector.mapping.max_model_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert_eq!(c.item_model_work(), expected); check_limits(&c).unwrap();
    c.detector.mapping.max_model_work.projected_logits = NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert_eq!(check_limits(&c).unwrap_err().code, BatchCode::InvalidLimits);
}
#[test]
fn configuration_bounds_both_stages_storage_and_every_native_axis() {
    for n in 0..5 {
        let mut c = config(); *axis(&mut c.detector.mapping.max_model_work, n) = 0;
        assert!(check_limits(&c).is_err());
        let mut c = config(); let mut item = c.item_model_work();
        *axis(&mut c.max_model_work, n) = *axis(&mut item, n) - 1;
        assert!(check_limits(&c).is_err());
    }
    let mut c = config(); c.detector.mapping.max_mask_visits = 1000;
    assert!(check_limits(&c).is_err()); c.request.verify = false; check_limits(&c).unwrap();
    c.detector.mapping.reduction.max_value_bytes = 1; assert!(check_limits(&c).is_err());
    let mut c = config(); c.detector.mapping.reduction.max_live_value_bytes = 0;
    assert!(check_limits(&c).is_err());
    let mut c = config(); c.detector.mapping.chunks.max_chunk_bytes = c.detector.planning.max_input_bytes + 1;
    assert!(check_limits(&c).is_err());
}
#[test]
fn corpus_reservation_is_atomic_across_all_five_axes_and_masks() {
    for n in 0..6 {
        let mut c = config(); let mut item = c.item_model_work();
        if n == 5 { c.max_mask_visits = 3999; }
        else { *axis(&mut c.max_model_work, n) = *axis(&mut item, n) - 1; }
        let mut ledger = Ledger::default();
        assert_eq!(ledger.begin(&c, context(1, 1)).unwrap_err().fault.code, BatchCode::WorkLimit);
        assert_eq!(ledger.reserved, Int4Work::default()); assert_eq!(ledger.masks, 0);
        assert_eq!(ledger.sequence, 0); assert!(ledger.failed); assert!(ledger.ready().is_err());
    }
}
#[test]
fn overflow_never_partially_charges_another_axis() {
    for n in 0..6 {
        let c = config(); let mut ledger = Ledger::default();
        if n == 5 { ledger.masks = u64::MAX; } else { *axis(&mut ledger.reserved, n) = u64::MAX; }
        let before = ledger.reserved; let masks = ledger.masks;
        assert!(ledger.begin(&c, context(1, 1)).is_err());
        assert_eq!(ledger.reserved, before); assert_eq!(ledger.masks, masks); assert_eq!(ledger.sequence, 0);
    }
}
#[test]
fn completed_early_output_does_not_refund_its_whole_document_allowance() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission(&c);
    for n in 1..=3 {
        let output = execute_reserved(&mut ledger, &c, context(n, 1), 8192, &mut a, &mut Control::default(),
            |_| (Ok("small-public-result"), true)).unwrap();
        drop(output); assert_eq!(ledger.masks, n * 4000);
    }
    assert_eq!(ledger.reserved, c.max_model_work);
    assert!(ledger.begin(&c, context(4, 1)).is_err());
}
#[test]
fn recoverable_failures_and_flush_epochs_keep_every_prior_reservation() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission(&c);
    let errors = [Int4SourceMapError::WorkLimit.into(), planning_error(SourcePlanningError::ContextBudget),
        Int4SourceMapError::Reduction(ExecutionError::LiveValueBudget).into()];
    for (index, error) in errors.into_iter().enumerate() {
        let n = index as u64 + 1;
        let failure = execute_reserved::<_, _, (), _>(&mut ledger, &c, context(n * 3, n), 8192, &mut a,
            &mut Control::default(), |_| (Err(error), true)).err().unwrap();
        assert!(!failure.stop); assert!(!ledger.failed); assert_eq!(ledger.masks, n * 4000);
    }
    assert_eq!(ledger.reserved, c.max_model_work); assert_eq!(ledger.sequence, 9); assert_eq!(ledger.epoch, 3);
    assert!(ledger.begin(&c, context(12, 4)).is_err());
}
#[test]
fn sequence_gaps_are_valid_but_replays_backward_epochs_and_lines_are_terminal() {
    for n in 0..5 {
        let c = config(); let mut ledger = Ledger::default();
        ledger.begin(&c, context(3, 2)).unwrap(); ledger.finish(&Ok::<(), BatchItemFailure>(()));
        let before = ledger.reserved;
        let mut next = context(7, 3);
        match n { 0 => next.request_seq = 3, 1 => next.epoch = 0, 2 => next.epoch = 1,
            3 => next.input_line = 0, _ => next.input_line = 6 }
        let error = ledger.begin(&c, next).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        assert_eq!(ledger.reserved, before); assert_eq!(ledger.masks, 4000);
    }
    let c = config(); let mut ledger = Ledger::default();
    ledger.begin(&c, context(3, 2)).unwrap(); ledger.finish(&Ok::<(), BatchItemFailure>(()));
    ledger.begin(&c, context(9, 4)).unwrap(); assert_eq!(ledger.masks, 8000);
}
#[test]
fn any_admitted_identity_substitution_stops_before_native_execution() {
    for n in 0..8 {
        let c = config(); let mut ledger = Ledger::default(); let mut a = admission(&c); a.drift = Some(n);
        let error = execute_reserved::<_, _, (), _>(&mut ledger, &c, context(1, 1), 8192, &mut a,
            &mut Control::default(), |_| panic!("must not execute after identity substitution")).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::Admission); assert!(error.stop); assert!(ledger.failed);
        assert_eq!(ledger.reserved, c.item_model_work()); assert_eq!(ledger.masks, 4000);
        assert_eq!(*a.trace.borrow(), ["admit", "guard"]);
    }
}
#[test]
fn a_rejected_permit_does_not_refund_the_reserved_attempt() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission(&c); a.refuse = true;
    let error = execute_reserved::<_, _, (), _>(&mut ledger, &c, context(1, 1), 8192, &mut a,
        &mut Control::default(), |_| panic!("admission refused")).err().unwrap();
    assert!(!error.stop); assert_eq!(ledger.reserved, c.item_model_work());
    assert_eq!(*a.trace.borrow(), ["admit"]);
    a.refuse = false;
    drop(execute_reserved(&mut ledger, &c, context(3, 2), 8192, &mut a, &mut Control::default(),
        |_| (Ok(()), true)).unwrap());
    assert_eq!(ledger.masks, 8000);
}
#[test]
fn output_storage_drops_before_its_admission_guard() {
    let c = config(); let mut ledger = Ledger::default(); let mut a = admission(&c); let trace = a.trace.clone();
    let output = execute_reserved(&mut ledger, &c, context(1, 1), 8192, &mut a, &mut Control::default(), |_| {
        (Ok(Body { text: "public-result".to_owned(), trace: trace.clone() }), true)
    }).unwrap();
    assert_eq!(*trace.borrow(), ["admit"]); drop(output);
    assert_eq!(*trace.borrow(), ["admit", "body", "guard"]); assert!(!ledger.failed);
}
#[test]
fn cancellation_before_admit_before_native_and_after_completion_never_exposes_a_body() {
    for stop in 1..=3 {
        let c = config(); let mut ledger = Ledger::default(); let mut a = admission(&c); let trace = a.trace.clone();
        let error = execute_reserved(&mut ledger, &c, context(1, 1), 8192, &mut a,
            &mut Control { calls: 0, stop_at: Some(stop) }, |_| {
                (Ok(Body { text: "private-completion".to_owned(), trace: trace.clone() }), true)
            }).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
        assert!(error.stop); assert!(ledger.failed); assert_eq!(ledger.reserved, c.item_model_work());
        let expected: &[&str] = match stop { 1 => &[], 2 => &["admit", "guard"], _ => &["admit", "body", "guard"] };
        assert_eq!(trace.borrow().as_slice(), expected);
    }
}
#[test]
fn dirty_native_state_makes_both_success_and_recoverable_failure_terminal() {
    let c = config();
    for success in [false, true] {
        let mut ledger = Ledger::default(); let mut a = admission(&c);
        let error = execute_reserved(&mut ledger, &c, context(1, 1), 8192, &mut a, &mut Control::default(), |_| {
            let output = if success { Ok("must-not-publish") } else { Err(Int4SourceMapError::WorkLimit.into()) };
            (output, false)
        }).err().unwrap();
        assert!(error.stop); assert!(ledger.failed); assert!(ledger.ready().is_err()); assert_eq!(ledger.masks, 4000);
    }
}
#[test]
fn nested_document_budgets_are_recoverable_fixed_protocol_errors() {
    let errors: Vec<(Int4LongRedactionError, BatchCode)> = vec![
        (Int4SourceMapError::EmptySource.into(), BatchCode::DocumentLimit),
        (Int4SourceMapError::Chunk(MapReduceError::InputBudget).into(), BatchCode::DocumentLimit),
        (Int4SourceMapError::Chunk(MapReduceError::ChunkBudget).into(), BatchCode::WorkLimit),
        (Int4SourceMapError::Chunk(MapReduceError::TokenBudget).into(), BatchCode::WorkLimit),
        (Int4SourceMapError::Chunk(MapReduceError::TokenizerBudget).into(), BatchCode::WorkLimit),
        (Int4SourceMapError::Reduction(ExecutionError::WorkBudget).into(), BatchCode::WorkLimit),
        (Int4SourceMapError::Reduction(ExecutionError::ValueBudget).into(), BatchCode::OutputLineLimit),
        (Int4SourceMapError::Reduction(ExecutionError::LiveValueBudget).into(), BatchCode::OutputLineLimit),
        (Int4SourceMapError::Reduction(ExecutionError::TotalValueBudget).into(), BatchCode::OutputLineLimit),
        (Int4SourceMapError::Reduction(ExecutionError::ResultBudget).into(), BatchCode::OutputLineLimit),
        (Int4SourceMapError::Reduction(ExecutionError::Task { stage: TaskStage::Map { first_chunk: 1 },
            source: Int4SourceError::Planning(SourcePlanningError::ContextBudget) }).into(), BatchCode::WorkLimit),
        (planning_error(SourcePlanningError::InputBudget), BatchCode::DocumentLimit),
        (planning_error(SourcePlanningError::ContextBudget), BatchCode::WorkLimit),
        (planning_error(SourcePlanningError::OutputBudget), BatchCode::OutputLineLimit),
        (planning_error(SourcePlanningError::Ner(NerError::OutputBudgetExceeded)), BatchCode::OutputLineLimit),
        (json_error(JsonDecodeError::BudgetExceeded("private-request-axis")), BatchCode::WorkLimit),
        (json_error(JsonDecodeError::Mask(MaskOracleError::WorkBudgetExceeded { max_trie_node_visits: 1 })), BatchCode::WorkLimit),
        (json_error(JsonDecodeError::NoLegalToken), BatchCode::Execution),
        (Int4RedactionError::Native(StrictInt4Error::Work).into(), BatchCode::WorkLimit),
        (Int4RedactionError::Redaction(RedactError::OutputBudget).into(), BatchCode::OutputLineLimit),
    ];
    for (error, code) in errors {
        let out = failure(error); assert!(!out.stop); assert_eq!(out.fault.code, code);
        let bytes = canonjson::canonical_bytes(&out.fault).unwrap();
        assert!(!String::from_utf8(bytes).unwrap().contains("private-request-axis"));
    }
}
#[test]
fn invalid_receipts_identity_and_dirty_invariants_are_never_recoverable() {
    let errors: Vec<Int4LongRedactionError> = vec![
        Int4SourceMapError::Admission.into(), Int4SourceMapError::InvalidLimits.into(),
        Int4SourceMapError::Chunk(MapReduceError::InvalidSpan).into(),
        Int4SourceMapError::Reduction(ExecutionError::Invariant).into(),
        source_error(Int4SourceError::InvalidResult),
        source_error(Int4SourceError::Extraction(Int4ExtractError::Identity)),
        source_error(Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::WorkMismatch))),
        source_error(Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::ModelIdentity))),
        planning_error(SourcePlanningError::Ner(NerError::InvalidResult)),
        json_error(JsonDecodeError::IndependentValidation), json_error(JsonDecodeError::EngineAlreadyPrimed),
        Int4RedactionError::Native(StrictInt4Error::EngineUnavailable).into(),
        Int4RedactionError::Native(StrictInt4Error::Cache).into(),
        Int4RedactionError::Redaction(RedactError::InvalidNerEvidence).into(),
        Int4RedactionError::Redaction(RedactError::KeyMismatch).into(),
    ];
    for error in errors { assert!(failure(error).stop); }
}
#[test]
fn typed_cancellation_survives_every_native_nesting_boundary() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast] {
        let errors = [
            Int4RedactionError::Cancelled(cause).into(),
            Int4RedactionError::Native(StrictInt4Error::Cancelled(cause)).into(),
            source_error(Int4SourceError::Cancelled(cause)),
            json_error(JsonDecodeError::Cancelled(cause)),
            planning_error(SourcePlanningError::Ner(NerError::Extraction(ExtractError::Decode(JsonDecodeError::Cancelled(cause))))),
            Int4SourceMapError::Reduction(ExecutionError::Task { stage: TaskStage::Map { first_chunk: 2 },
                source: Int4SourceError::Cancelled(cause) }).into(),
        ];
        for error in errors {
            let error = failure(error); assert!(error.stop);
            assert_eq!(error.fault.code, BatchCode::Cancelled); assert_eq!(error.fault.cancellation, Some(cause));
        }
    }
}
fn receipt(c: &Int4DocumentRedactionBatchConfig, pseudonyms: Option<&Pseudonyms<'_>>) -> Int4LongRedactionRun {
    let source = "alpha beta";
    let mut document = DetectedDocument::rules_only(source, &c.request.rules, c.request.rule_budget).unwrap();
    document.model_types = c.detector.ner.types.iter().copied().collect();
    let mut result = actions::apply(&document, &c.request.actions, pseudonyms, c.request.edit_budget).unwrap();
    if c.request.verify { result.verification = VerificationStatus::CleanDeclaredUnion; }
    pipeline::bind_request(&mut result, &c.request).unwrap();
    let one = constrained_int4::planned_work(WIDTH, 128, 16).unwrap(); let two = one.checked_add(one).unwrap();
    let used = SourceMapVerification { fields: 2, matches: 3, scan_steps: 7 };
    let stage = || Int4LongRedactionStage { preflight: Int4LongRedactionPreflight {
        source_bytes: source.len(), source_scalars: source.chars().count(), chunks: 2, group_width: WIDTH,
        planned_model_work: two, reserved_mask_node_visits: 2000 }, model_work: two, mask_node_visit_charge: 4,
        map_verification_used: used, detection_verification_used: used };
    let work = if c.request.verify { two.checked_add(two).unwrap() } else { two };
    let total = if c.request.verify { add_verification(used, used).unwrap() } else { used };
    Int4LongRedactionRun { schema_version: 1, execution: INT4_LONG_REDACTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: WIDTH, detector_scope: "whole-source-rules-independent-ner-chunks-v1",
        result, original: stage(), verification: c.request.verify.then(stage), reserved_model_work: work, model_work: work,
        reserved_mask_node_visits: if c.request.verify { 4000 } else { 2000 },
        mask_node_visit_charge: if c.request.verify { 8 } else { 4 },
        map_verification_used: total, detection_verification_used: total, verification_scan_steps: total.scan_steps,
        warnings: ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
            "clean_scan_and_pseudonyms_are_not_anonymization"] }
}
#[test]
fn completed_receipt_reconciles_geometry_policy_group_and_every_counter() {
    let c = config(); let original = receipt(&c, None).original.preflight;
    validate_result(&receipt(&c, None), original, &c, WIDTH, None).unwrap();
    for n in 0..44 {
        let mut out = receipt(&c, None);
        match n { 0 => out.original.preflight.source_bytes += 1,
            1 => out.verification.as_mut().unwrap().preflight.source_bytes += 1,
            2 => out.verification.as_mut().unwrap().preflight.source_scalars += 1,
            3 => out.verification.as_mut().unwrap().preflight.chunks += 1,
            4 => out.original.preflight.group_width = 32, 5 => out.verification.as_mut().unwrap().preflight.group_width = 32,
            6 => out.group_width = 32, 7 => out.numerics_profile = "strict-quantized-v1",
            8 => out.execution = "other", 9 => out.detector_scope = "rules-only", 10 => out.warnings[0] = "unearned-quality",
            11 => out.schema_version = 2, 12 => out.result.schema_version = 2, 13 => out.result.model_types.clear(),
            14 => out.result.rules.enabled.clear(), 15 => out.result.rule_profile = "other".to_owned(),
            16 => out.result.action_policy = "other".to_owned(), 17 => out.result.overlap_policy = "other".to_owned(),
            18 => out.result.task_spec_version = "other".to_owned(), 19 => out.result.verification = VerificationStatus::NotRequested,
            20 => out.verification = None,
            21..=25 => *axis(&mut out.reserved_model_work, n - 21) += 1,
            26..=30 => *axis(&mut out.model_work, n - 26) += 1,
            31 => out.reserved_mask_node_visits += 1, 32 => out.mask_node_visit_charge += 1,
            33 => out.map_verification_used.fields += 1, 34 => out.map_verification_used.matches += 1,
            35 => out.map_verification_used.scan_steps += 1, 36 => out.detection_verification_used.fields += 1,
            37 => out.detection_verification_used.matches += 1, 38 => out.detection_verification_used.scan_steps += 1,
            39 => out.verification_scan_steps += 1, 40 => out.original.map_verification_used.fields += 1,
            41 => out.original.detection_verification_used.scan_steps += 1, 42 => out.original.model_work = Int4Work::default(),
            _ => out.original.mask_node_visit_charge = 2001 }
        assert!(validate_result(&out, original, &c, WIDTH, None).is_err(), "{n}");
    }
}
#[test]
fn a_stage_cannot_borrow_native_allowance_from_the_other_stage() {
    let c = config();
    for n in 0..5 {
        let mut out = receipt(&c, None); let original = out.original.preflight;
        let step = if n == 1 { NANBEIGE_VOCAB_SIZE as u64 } else { 1 };
        *axis(&mut out.original.model_work, n) += step;
        *axis(&mut out.verification.as_mut().unwrap().model_work, n) -= step;
        assert!(fits(out.model_work, out.reserved_model_work));
        assert!(validate_result(&out, original, &c, WIDTH, None).is_err(), "{n}");
    }
}
#[test]
fn each_grounding_pool_is_cumulative_and_cannot_borrow_from_the_other() {
    for n in 0..6 {
        let mut c = config(); let out = receipt(&c, None); let original = out.original.preflight;
        let budget = if n < 3 { &mut c.detector.mapping.verification } else { &mut c.request.grounding_budget };
        match n % 3 { 0 => budget.max_fields = 3, 1 => budget.max_matches = 5, _ => budget.max_scan_steps = 13 }
        assert!(validate_result(&out, original, &c, WIDTH, None).is_err(), "{n}");
    }
    let zero = SourceMapVerification { fields: 0, matches: 0, scan_steps: 0 };
    for n in 0..3 {
        let mut a = zero; let mut b = zero;
        match n { 0 => { a.fields = usize::MAX; b.fields = 1; },
            1 => { a.matches = usize::MAX; b.matches = 1; }, _ => { a.scan_steps = u64::MAX; b.scan_steps = 1; } }
        assert!(add_verification(a, b).is_none());
    }
}
#[test]
fn unverified_output_must_omit_its_second_stage_and_name_its_status() {
    let mut c = config(); c.request.verify = false;
    let out = receipt(&c, None); let original = out.original.preflight;
    validate_result(&out, original, &c, WIDTH, None).unwrap();
    assert!(out.verification.is_none()); assert_eq!(out.result.verification(), VerificationStatus::NotRequested);
    let mut corrupt = receipt(&c, None); corrupt.result.verification = VerificationStatus::CleanDeclaredUnion;
    assert!(validate_result(&corrupt, original, &c, WIDTH, None).is_err());
}
#[test]
fn fixed_key_and_namespace_commitments_survive_epochs_and_cannot_be_substituted() {
    let c = config(); let key = PseudonymKey::from_bytes(&[0x63; 32], "opaque-key").unwrap();
    let same = Pseudonyms::full256(&key, "private-corpus-namespace", None).unwrap();
    let other_scope = Pseudonyms::full256(&key, "different-namespace", None).unwrap();
    let other_key = PseudonymKey::from_bytes(&[0x64; 32], "opaque-key").unwrap();
    let other_secret = Pseudonyms::full256(&other_key, "private-corpus-namespace", None).unwrap();
    let out = receipt(&c, Some(&same)); let original = out.original.preflight;
    validate_result(&out, original, &c, WIDTH, Some(&same)).unwrap();
    for different in [None, Some(&other_scope), Some(&other_secret)] {
        assert!(validate_result(&out, original, &c, WIDTH, different).is_err());
    }
    let first = same.pseudonym(PiiKind::Person, "private-name").unwrap();
    let second = same.pseudonym(PiiKind::Person, "private-name").unwrap(); assert_eq!(first, second);
    assert_ne!(first, other_scope.pseudonym(PiiKind::Person, "private-name").unwrap());
    let serialized = String::from_utf8(canonjson::canonical_bytes(&out).unwrap()).unwrap();
    for private in ["private-corpus-namespace", "private-name", "generated_token_ids", "prompt_tokens", "raw_source"] {
        assert!(!serialized.contains(private));
    }
}
#[test]
fn grouped_work_is_an_independent_context_envelope_not_one_concatenated_sequence() {
    let one = constrained_int4::planned_work(WIDTH, 128, 16).unwrap();
    let total = one.checked_add(one).unwrap();
    let concatenated = Int4Work::for_sequence(WIDTH, 0, total.forward_positions as usize, total.projected_logits as usize).unwrap();
    assert_ne!(total.attention_pairs, concatenated.attention_pairs);
    let other = constrained_int4::planned_work(32, 128, 16).unwrap();
    assert_ne!(one.projections.dot_products, other.projections.dot_products);
    let c = config(); let out = receipt(&c, None);
    assert!(validate_result(&out, out.original.preflight, &c, 32, None).is_err());
}
#[test]
fn residual_reports_never_reach_corpus_error_output() {
    let report = pipeline::LeakReport { schema_version: 1, residuals: Vec::new(),
        rules: Default::default(), model_types: [EntityType::Person].into_iter().collect() };
    let error = failure(Int4RedactionError::Residual(report).into());
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Execution);
    let text = String::from_utf8(canonjson::canonical_bytes(&error.fault).unwrap()).unwrap();
    for field in ["residuals", "rules", "model_types", "text", "pseudonym"] { assert!(!text.contains(field)); }
}
