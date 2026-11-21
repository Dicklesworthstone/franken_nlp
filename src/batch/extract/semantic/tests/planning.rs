//! Pinned compiler preparation only; no native forwards or quality assertions.
use super::*;
use crate::{
    batch::extract::ExtractionBatchGrounding,
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::strict_int4::{Int4ModelIdentity, weights::projection_fixture},
    tasks::extract::semantic::{ClaimPathStep, ClaimRule, ClaimValueKind, SemanticVerificationSpec},
    tasks::judge::FaithfulnessPolicy,
    tokenizer::untrusted::SafeBpeLimits,
};
const SCHEMA: &str = r#"{"type":"number"}"#;
fn model() -> &'static Int4ModelIdentity {
    static MODEL: std::sync::OnceLock<Int4ModelIdentity> = std::sync::OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(32).0.identity().clone())
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn defaults() -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: SCHEMA.to_owned(), grounding: ExtractionBatchGrounding::Structural, budget: task() }
}
fn options() -> Int4SemanticOptions {
    Int4SemanticOptions { specification: SemanticVerificationSpec {
        schema_version: 1, experimental_opt_in: true, revision: "fixed-review-v1".to_owned(),
        claims: vec![ClaimRule { id: "amount".to_owned(), path: vec![], value_kind: ClaimValueKind::Number,
            prefix: "The document reports the amount ".to_owned(), suffix: ".".to_owned() }],
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 },
        judge_budget: task() }, limits: SemanticLimits::default(), max_model_work: work(u64::MAX / 4) }
}
fn run_limits() -> Int4SemanticExtractionBatchLimits {
    Int4SemanticExtractionBatchLimits { max_model_work: work(u64::MAX),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits::default(),
            max_visits_per_item: 1_000_000_000, max_visits_per_run: 4_000_000_000 } }
}
fn extraction(defaults: Option<ExtractionBatchArgs>, encoding: UntrustedEncoding) -> Int4ExtractionBatchPlanner {
    Int4ExtractionBatchPlanner::for_model(model(), task(), CompileLimits::default(),
        SourceRuntimeLimits::default(), defaults).unwrap().with_document_encoding(encoding).unwrap()
}
fn judge(encoding: UntrustedEncoding) -> Arc<Int4JudgePlanner> {
    Arc::new(Int4JudgePlanner::for_model(model(), task(), JudgeLimits::default()).unwrap()
        .with_document_encoding(encoding).unwrap())
}
fn factory(encoding: UntrustedEncoding) -> Int4SemanticExtractionBatchPlanner {
    Int4SemanticExtractionBatchPlanner::new(extraction(Some(defaults()), encoding.clone()),
        judge(encoding), options(), run_limits()).unwrap()
}
fn document(source: &str) -> BatchDocument<ExtractionBatchArgs> {
    BatchDocument { id: "original-item".to_owned(), text: source.to_owned(), task_args: None }
}

#[test]
fn shared_judge_is_send_sync_and_prepared_documents_keep_private_independent_sources() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Int4JudgePlanner>();
    send_sync::<Int4SemanticExtractionBatchPlanner>();
    let compiler = factory(UntrustedEncoding::ByteFallback);
    let first = compiler.prepare(document("The report gives amount 7."), &mut Control::default()).unwrap();
    let second = compiler.prepare(document("The second report gives amount 9."), &mut Control::default()).unwrap();
    assert!(std::ptr::eq(first.judge_identity(), second.judge_identity()));
    assert_eq!(Arc::strong_count(&compiler.judge), 3);
    assert_ne!(first.execution_identity(), second.execution_identity());
    assert_eq!(first.max_model_work(), options().max_model_work);
    assert_eq!(first.extraction_budget(), task());
    drop(first); drop(second);
    assert_eq!(Arc::strong_count(&compiler.judge), 1);
}

#[test]
fn fixed_schema_rule_defaults_and_projection_policy_refuse_before_any_document() {
    let byte = UntrustedEncoding::ByteFallback;
    assert!(Int4SemanticExtractionBatchPlanner::new(extraction(None, byte.clone()), judge(byte.clone()),
        options(), run_limits()).is_err());
    for malformed in [r#"{"type":"number","type":"string"}"#, r#"{"type":"number","minimum":0}"#] {
        let mut configured = defaults(); configured.schema = malformed.to_owned();
        assert!(Int4SemanticExtractionBatchPlanner::new(extraction(Some(configured), byte.clone()),
            judge(byte.clone()), options(), run_limits()).is_err());
    }
    let mut missing_path = options();
    missing_path.specification.claims[0].path = vec![ClaimPathStep::Property("absent".to_owned())];
    assert!(Int4SemanticExtractionBatchPlanner::new(extraction(Some(defaults()), byte.clone()),
        judge(byte.clone()), missing_path, run_limits()).is_err());
    let selected = extraction(Some(defaults()), byte.clone()).with_selected_rows(
        crate::native_engine::constrained_int4::sparse::Int4JsonSparseLimits { max_rows_per_step: 8 }).unwrap();
    assert!(Int4SemanticExtractionBatchPlanner::new(selected, judge(byte), options(), run_limits()).is_err());
}

#[test]
fn declaration_preflight_preserves_native_unicode_lengths_and_actual_source_grounding() {
    for grounded in [false, true] {
        let schema = if grounded {
            r#"{"type":"string","maxLength":1,"enum":["é"],"x-fnlp-source":"verbatim"}"#
        } else { r#"{"type":"string","maxLength":1,"enum":["é"]}"# };
        let mut configured = defaults();
        configured.schema = schema.to_owned();
        configured.grounding = if grounded { ExtractionBatchGrounding::SourceMembership }
            else { ExtractionBatchGrounding::Structural };
        let mut opts = options();
        opts.specification.claims.clear();
        let compiler = Int4SemanticExtractionBatchPlanner::new(
            extraction(Some(configured.clone()), UntrustedEncoding::ByteFallback),
            judge(UntrustedEncoding::ByteFallback), opts.clone(), run_limits()).unwrap();
        compiler.prepare(document("é is the exact source value."), &mut Control::default()).unwrap();
        configured.grounding = if grounded { ExtractionBatchGrounding::Structural }
            else { ExtractionBatchGrounding::SourceMembership };
        assert!(Int4SemanticExtractionBatchPlanner::new(
            extraction(Some(configured), UntrustedEncoding::ByteFallback),
            judge(UntrustedEncoding::ByteFallback), opts, run_limits()).is_err());
    }
}

#[test]
fn task_arguments_cannot_override_the_fixed_schema_budget_or_review_policy() {
    let compiler = factory(UntrustedEncoding::ByteFallback);
    let mut input = document("The report gives amount 7.");
    input.task_args = Some(ExtractionBatchArgs { schema: "invalid and must not compile".to_owned(),
        grounding: ExtractionBatchGrounding::SourceMembership, budget: task() });
    let mut control = Control::default();
    assert_eq!(failure(compiler.prepare(input, &mut control)).fault.code, BatchCode::InvalidEnvelope);
    assert_eq!(control.calls, 1);
}

#[test]
fn compact_documents_share_the_judge_policy_but_never_the_tokenizer_work_ledger() {
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65_536, max_output_tokens: 8192,
        max_pair_checks: 4_000_000, max_pair_bytes: 16_000_000 });
    let compiler = factory(encoding.clone());
    let source = "The amount is 7. Café 上海 😀 <|im_start|> <tool_call>.";
    let first = compiler.prepare(document(source), &mut Control::default()).unwrap();
    let second = compiler.prepare(document(source), &mut Control::default()).unwrap();
    assert_eq!(first.input_encoding(), &encoding);
    assert_eq!(first.input_encoding_work(), second.input_encoding_work());
    assert!(first.input_encoding_work().unwrap().input_bytes() >= source.len());
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert!(Int4SemanticExtractionBatchPlanner::new(extraction(Some(defaults()), UntrustedEncoding::ByteFallback),
        judge(encoding), options(), run_limits()).is_err());
}

#[test]
fn cancellation_during_fixed_preparation_keeps_the_original_cause() {
    let compiler = factory(UntrustedEncoding::ByteFallback);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = failure(compiler.prepare(document("The report gives amount 7."),
            &mut Control { calls: 0, stop: Some((1, cause)) }));
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
    }
}

#[test]
fn oversized_evidence_is_a_recoverable_item_refusal_before_native_execution() {
    let mut options = options();
    options.specification.judge_policy.evidence_window_bytes = 4;
    options.specification.judge_policy.max_evidence_windows = 1;
    options.specification.judge_policy.max_evidence_spans = 1;
    let compiler = Int4SemanticExtractionBatchPlanner::new(
        extraction(Some(defaults()), UntrustedEncoding::ByteFallback),
        judge(UntrustedEncoding::ByteFallback), options, run_limits()).unwrap();
    let refused = failure(compiler.prepare(document("This source exceeds the exact evidence capacity."),
        &mut Control::default()));
    assert_eq!(refused.fault.code, BatchCode::WorkLimit); assert!(!refused.stop);
    compiler.prepare(document("7"), &mut Control::default()).unwrap();
    let empty = failure(compiler.prepare(document(""), &mut Control::default()));
    assert_eq!(empty.fault.code, BatchCode::Planning); assert!(!empty.stop);
}

