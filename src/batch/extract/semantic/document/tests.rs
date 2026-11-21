//! Model-free factory, authority and accounting regressions. The tiny weights
//! fixture supplies an authentic typed model identity, never native inference
//! results, quality evidence or a parity claim.
use super::*;
use crate::{
    batch::extract::grouped_int4::{Int4ExtractionBatchPlanner,
        long::{Int4ExtractionMapLimits, Int4ExtractionMappingLimits}},
    grammar::runtime::SourceRuntimeLimits,
    native_engine::{portable_int8::ProjectionWork,
        strict_int4::{Int4ModelIdentity, weights::projection_fixture}},
    tasks::{extract::{object_reconcile::RootObjectLimits,
            semantic::{ClaimPathStep, ClaimRule, ClaimValueKind, SemanticVerificationSpec}},
        judge::{FaithfulnessAggregationPolicy, FaithfulnessPolicy},
        mapreduce::{ChunkLimits, ExecutionLimits}},
    tokenizer::untrusted::SafeBpeLimits,
};

const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"amount":{"type":"number"},"name":{"type":"string","maxLength":16}},"required":["amount","name"]}"#;
const SOURCE: &str = "The amount is 7. The name is Alice. Café 上海 😀 <think> stays literal.";

fn model() -> &'static Int4ModelIdentity {
    static MODEL: std::sync::OnceLock<Int4ModelIdentity> = std::sync::OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(32).0.identity().clone())
}
fn work(value: u64) -> Int4Work {
    Int4Work { forward_positions: value, projected_logits: value, attention_pairs: value,
        projections: ProjectionWork { dot_products: value, multiply_accumulates: value } }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn request() -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: SCHEMA.to_owned(), grounding: ExtractionBatchGrounding::Structural, budget: task() }
}
fn options() -> Int4SemanticOptions {
    Int4SemanticOptions { specification: SemanticVerificationSpec {
        schema_version: 1, experimental_opt_in: true, revision: "object-population-review-v1".to_owned(),
        claims: vec![ClaimRule { id: "amount".to_owned(),
            path: vec![ClaimPathStep::Property("amount".to_owned())], value_kind: ClaimValueKind::Number,
            prefix: "The document reports the amount ".to_owned(), suffix: ".".to_owned() }],
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 16, max_evidence_windows: 8, max_evidence_spans: 8 },
        judge_budget: task() }, limits: SemanticLimits::default(), max_model_work: work(u64::MAX / 4) }
}
fn extraction_limits() -> Int4ObjectExtractionMapLimits {
    Int4ObjectExtractionMapLimits {
        mapping: Int4ExtractionMapLimits {
            mapping: Int4ExtractionMappingLimits {
                chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 16,
                    max_chunk_tokens: 8192, context_tokens: 8192, reserved_tokens: 64,
                    max_chunks: 64, max_tokenizer_calls: 4096 },
                reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 2,
                    max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
                max_model_work: work(u64::MAX), mask_limits: MaskWorkLimits::default(),
                mask_visits_per_chunk: 10_000, max_mask_visits: 640_000,
            },
            verification: GroundingBudget::default(), root_array_union: None,
        },
        reconciliation: RootObjectLimits { max_properties: 32, max_origins: 4096,
            max_input_bytes: 4 << 20, max_json_bytes: 1 << 20,
            max_nodes: 65536, max_result_bytes: 2 << 20 },
    }
}
fn review_limits() -> Int4FaithfulnessMapLimits {
    Int4FaithfulnessMapLimits {
        chunks: ChunkLimits { max_input_bytes: 8192 + options().limits.max_claim_bytes,
            max_chunk_bytes: 16, max_chunk_tokens: 8192, context_tokens: 8192,
            reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 4096 },
        max_model_work: work(u64::MAX), max_retained_result_bytes: 4 << 20,
        max_result_bytes: 8 << 20, aggregation: FaithfulnessAggregationPolicy::default(),
    }
}
fn limits() -> Int4SemanticObjectDocumentBatchLimits {
    let per = extraction_limits().mapping.verification;
    Int4SemanticObjectDocumentBatchLimits { max_model_work: work(u64::MAX),
        max_mask_node_visits: extraction_limits().mapping.mapping.max_mask_visits * 3,
        max_verification: GroundingBudget { max_fields: per.max_fields * 3,
            max_matches: per.max_matches * 3, max_scan_steps: per.max_scan_steps * 3 } }
}
fn compact() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 26, max_pair_bytes: 1 << 29 }
}
fn factory_with(encoding: UntrustedEncoding, request: ExtractionBatchArgs, options: Int4SemanticOptions)
    -> Result<Int4SemanticObjectDocumentBatchPlanner, BatchFault> {
    let extraction = Int4ExtractionBatchPlanner::for_model(model(), task(), CompileLimits::default(),
        SourceRuntimeLimits::default(), None)?.with_document_encoding(encoding.clone())?;
    let extraction = Int4ObjectExtractionPlanner::new(extraction).map_err(|_| BatchCode::Planning)?;
    let judge = Int4JudgePlanner::for_model(model(), task(), JudgeLimits::default()).unwrap()
        .with_document_encoding(encoding).unwrap();
    Int4SemanticObjectDocumentBatchPlanner::new(extraction, request, Arc::new(judge), options,
        extraction_limits(), review_limits(), limits())
}
fn factory(encoding: UntrustedEncoding) -> Int4SemanticObjectDocumentBatchPlanner {
    factory_with(encoding, request(), options()).unwrap()
}
fn document(source: &str) -> BatchDocument<ExtractionDocumentArgs> {
    BatchDocument { id: "original-item".to_owned(), text: source.to_owned(), task_args: None }
}
fn failure<T>(value: Result<T, BatchItemFailure>) -> BatchItemFailure {
    match value { Ok(_) => panic!("expected a refusal"), Err(error) => error }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
#[derive(Default)]
struct Control { calls: usize, stop: Option<(usize, DecodeCancellationKind)> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        self.stop.and_then(|(call, cause)| (self.calls >= call).then_some(cause))
    }
}

#[test]
fn complete_original_sources_and_exact_required_schema_stay_bound_to_the_factory() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Int4SemanticObjectDocumentBatchPlanner>();
    let compiler = factory(UntrustedEncoding::ByteFallback);
    let prepared = compiler.prepare(document(SOURCE), &mut Continue).unwrap();
    assert_eq!(prepared.source, SOURCE);
    assert!(prepared.chunk_count() > 1);
    assert_eq!(prepared.execution_identities().len(), prepared.chunk_count());
    assert_eq!(prepared.model_work(), options().max_model_work);
    assert_eq!(prepared.mask_node_visits(), extraction_limits().mapping.mapping.max_mask_visits);
    assert_eq!(prepared.verification_budget(), extraction_limits().mapping.verification);
    assert_eq!(prepared.max_result_bytes(), options().limits.max_output_bytes);
    assert_eq!(compiler.request().schema, SCHEMA);
    let same = compiler.plan(SOURCE, &mut Continue, false).unwrap();
    compiler.compare(&prepared, &same, &mut Continue).unwrap();
    assert!(prepared.expected == same.preflight_metadata());
    let changed = SOURCE.replace("Alice", "Elise");
    assert_eq!(changed.len(), SOURCE.len());
    let changed = compiler.plan(&changed, &mut Continue, false).unwrap();
    assert_eq!(failure(compiler.compare(&prepared, &changed, &mut Continue)).fault.code, BatchCode::Admission);
    let other = factory(UntrustedEncoding::ByteFallback);
    assert_eq!(failure(other.verify_prepared(&prepared)).fault.code, BatchCode::Admission);
    assert_eq!(Arc::strong_count(&compiler.judge), 3);
}

#[test]
fn changed_original_required_properties_cannot_reuse_identical_partial_schemas() {
    let first = factory(UntrustedEncoding::ByteFallback);
    let prepared = first.prepare(document(SOURCE), &mut Continue).unwrap();
    let mut changed = request();
    changed.schema = SCHEMA.replace(r#""required":["amount","name"]"#, r#""required":["amount"]"#);
    let second = factory_with(UntrustedEncoding::ByteFallback, changed, options()).unwrap();
    let planned = second.plan(SOURCE, &mut Continue, false).unwrap();
    // Per-chunk partial declarations may match; original-schema authority must not.
    assert!(prepared.expected != planned.preflight_metadata());
    assert!(failure(second.verify_prepared(&prepared)).stop);
}

#[test]
fn fixed_schema_claims_grounding_and_opt_in_refuse_before_any_population() {
    for invalid in [r#"{"type":"array","items":{"type":"number"},"maxItems":4}"#,
        r#"{"type":"object","type":"number","additionalProperties":false,"properties":{}}"#] {
        let mut changed = request(); changed.schema = invalid.to_owned();
        assert!(factory_with(UntrustedEncoding::ByteFallback, changed, options()).is_err());
    }
    let mut selected = options(); selected.specification.experimental_opt_in = false;
    assert!(factory_with(UntrustedEncoding::ByteFallback, request(), selected).is_err());
    let mut selected = options();
    selected.specification.claims[0].path = vec![ClaimPathStep::Property("absent".to_owned())];
    assert!(factory_with(UntrustedEncoding::ByteFallback, request(), selected).is_err());
    let mut changed = request(); changed.grounding = ExtractionBatchGrounding::SourceMembership;
    assert!(factory_with(UntrustedEncoding::ByteFallback, changed, options()).is_err());
}

#[test]
fn record_arguments_refuse_before_source_encoding_and_empty_records_do_not_spend_execution() {
    let compiler = factory(UntrustedEncoding::ByteFallback);
    let mut input = document(SOURCE); input.task_args = Some(ExtractionDocumentArgs {});
    let mut control = Control::default();
    let error = failure(compiler.prepare(input, &mut control));
    assert_eq!(error.fault.code, BatchCode::InvalidEnvelope); assert!(!error.stop);
    assert_eq!(control.calls, 1);
    let empty = failure(compiler.prepare(document(""), &mut Continue));
    assert_eq!(empty.fault.code, BatchCode::DocumentLimit); assert!(!empty.stop);
    compiler.prepare(document(SOURCE), &mut Continue).unwrap();
    for invalid in [r#"{"schema":"anything"}"#, r#"{"review":{}}"#, "[]", "true"] {
        assert!(serde_json::from_str::<ExtractionDocumentArgs>(invalid).is_err());
    }
}

#[test]
fn every_typed_cancel_reason_survives_planning_and_exact_reconstruction() {
    let compiler = factory(UntrustedEncoding::ByteFallback);
    let prepared = compiler.prepare(document(SOURCE), &mut Continue).unwrap();
    let plan = compiler.plan(SOURCE, &mut Continue, false).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop_at in [1, 4, 20] {
            let mut control = Control { calls: 0, stop: Some((stop_at, cause)) };
            let error = failure(compiler.prepare(document(SOURCE), &mut control));
            assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
        }
        let mut control = Control { calls: 0, stop: Some((1, cause)) };
        let error = failure(compiler.compare(&prepared, &plan, &mut control));
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
    }
}

#[test]
fn complete_attempt_caps_are_atomic_on_all_nine_axes_and_do_not_renew_at_flush() {
    let mut compiler = factory(UntrustedEncoding::ByteFallback);
    let per = compiler.options.max_model_work;
    let mask = compiler.extraction_limits.mapping.mapping.max_mask_visits;
    let grounding = compiler.extraction_limits.mapping.verification;
    for axis in 0..9 {
        compiler.limits = limits();
        match axis {
            0 => compiler.limits.max_model_work.forward_positions = per.forward_positions,
            1 => compiler.limits.max_model_work.projected_logits = per.projected_logits,
            2 => compiler.limits.max_model_work.attention_pairs = per.attention_pairs,
            3 => compiler.limits.max_model_work.projections.dot_products = per.projections.dot_products,
            4 => compiler.limits.max_model_work.projections.multiply_accumulates = per.projections.multiply_accumulates,
            5 => compiler.limits.max_mask_node_visits = mask,
            6 => compiler.limits.max_verification.max_fields = grounding.max_fields,
            7 => compiler.limits.max_verification.max_matches = grounding.max_matches,
            _ => compiler.limits.max_verification.max_scan_steps = grounding.max_scan_steps,
        }
        compiler.limits.validate_item(per, mask, grounding).unwrap();
        let mut ledger = Ledger::default();
        ledger.begin(&compiler, context(1, 1)).unwrap();
        assert_eq!(ledger.work, per); assert_eq!(ledger.masks, mask);
        assert_eq!(ledger.verification, grounding); assert_eq!(ledger.sequence, 1);
        // Test the successful-attempt transition explicitly. A later numbered
        // epoch grants no fresh native, mask or original-source scan budget.
        ledger.failed = false;
        let error = ledger.begin(&compiler, context(3, 2)).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::WorkLimit, "axis {axis}");
        assert!(error.stop); assert!(ledger.failed);
        assert_eq!(ledger.work, per); assert_eq!(ledger.masks, mask);
        assert_eq!(ledger.verification, grounding); assert_eq!(ledger.sequence, 1);
        assert_eq!(ledger.ready().unwrap_err().fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn cancellation_unwind_or_callback_failure_keeps_the_entire_debit_and_closed_latch() {
    let compiler = factory(UntrustedEncoding::ByteFallback);
    let mut ledger = Ledger::default();
    ledger.begin(&compiler, context(1, 1)).unwrap();
    // begin closes the latch before all fallible reconstruction/admission/native
    // work. Only the complete verified-result path can reopen it.
    let charged = (ledger.work, ledger.masks, ledger.verification);
    assert!(ledger.failed);
    let error = ledger.begin(&compiler, context(2, 2)).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    assert_eq!((ledger.work, ledger.masks, ledger.verification), charged);
}

#[test]
fn request_sequence_and_counter_overflow_cannot_create_execution_authority() {
    let compiler = factory(UntrustedEncoding::ByteFallback);
    for invalid in [context(0, 1), context(1, 0),
        BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut ledger = Ledger::default();
        assert_eq!(ledger.begin(&compiler, invalid).unwrap_err().fault.code, BatchCode::InvalidExecution);
        assert_eq!(ledger.work, Int4Work::default()); assert_eq!(ledger.masks, 0);
        assert!(ledger.failed);
    }
    let mut ledger = Ledger::default();
    ledger.begin(&compiler, context(1, 1)).unwrap(); ledger.failed = false;
    assert_eq!(ledger.begin(&compiler, context(1, 2)).unwrap_err().fault.code, BatchCode::InvalidExecution);
    let mut ledger = Ledger { work: work(u64::MAX), ..Ledger::default() };
    assert_eq!(ledger.begin(&compiler, context(1, 1)).unwrap_err().fault.code, BatchCode::WorkLimit);
    assert_eq!(ledger.masks, 0); assert_eq!(ledger.sequence, 0);
    let mut ledger = Ledger { verification: GroundingBudget {
        max_fields: usize::MAX, max_matches: 0, max_scan_steps: 0 }, ..Ledger::default() };
    assert_eq!(ledger.begin(&compiler, context(1, 1)).unwrap_err().fault.code, BatchCode::WorkLimit);
    assert_eq!(ledger.work, Int4Work::default()); assert_eq!(ledger.masks, 0);
}

#[test]
fn compact_reconstruction_cannot_renew_any_of_the_four_tokenizer_axes() {
    let compiler = factory(UntrustedEncoding::SafeBpeV1(compact()));
    let prepared = compiler.prepare(document(SOURCE), &mut Continue).unwrap();
    let used = prepared.encoding_work;
    assert!(used.input_bytes() > SOURCE.len());
    assert!(used.output_tokens() > 0); assert!(used.pair_checks() > 0); assert!(used.pair_bytes() > 0);
    compiler.admit_reconstruction(&prepared).unwrap();
    let repeated = compiler.plan(SOURCE, &mut Continue, false).unwrap();
    compiler.compare(&prepared, &repeated, &mut Continue).unwrap();
    let repeated = repeated.with_prior_preparation_encoding_work(used).unwrap();
    assert_eq!(repeated.input_encoding_work(), used.checked_add(used));
    assert!(repeated.preflight_metadata() != prepared.expected);
    // Fixed encoding policy changes identity, so construct and prepare an
    // actual new factory at each exact one-pass cap. Preparation fits, but the
    // repeat cost refuses BEFORE reconstruction or any native forward.
    for axis in 0..4 {
        let mut cap = compact();
        match axis { 0 => cap.max_input_bytes = used.input_bytes(),
            1 => cap.max_output_tokens = used.output_tokens(),
            2 => cap.max_pair_checks = used.pair_checks(),
            _ => cap.max_pair_bytes = used.pair_bytes() }
        let bounded = factory(UntrustedEncoding::SafeBpeV1(cap));
        let item = bounded.prepare(document(SOURCE), &mut Continue).unwrap();
        assert_eq!(item.encoding_work, used);
        let refused = failure(bounded.admit_reconstruction(&item));
        assert_eq!(refused.fault.code, BatchCode::WorkLimit, "encoding axis {axis}");
        assert!(refused.stop);
    }
}
