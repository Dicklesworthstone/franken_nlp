//! Real pinned planning with a private driver used only to reject native
//! attempts. Numerical/result parity remains the complete-map test authority.
use super::*;
use std::sync::{OnceLock, atomic::{AtomicUsize, Ordering}};
use crate::{
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind,
        lmhead::scoring::{ScoringMode, SequenceScoreRule},
        strict_int8::STRICT_INT8_EXECUTION},
    tasks::{mapreduce::ChunkLimits, sentiment::{SentimentAggregationPolicy, SentimentOptions, SentimentPolicy}},
    tokenizer::pinned_controls,
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 4 * 2048 * KV_BYTES_PER_TOKEN as u64 }
}
fn work() -> Int8Work {
    Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
        projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } }
}
fn mapping() -> Int8SentimentMapLimits {
    Int8SentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 16_384, max_chunk_bytes: 8,
        max_chunk_tokens: 1024, context_tokens: 2048, reserved_tokens: 1,
        max_chunks: 256, max_tokenizer_calls: 8192 }, max_model_work: work(),
        max_retained_result_bytes: 4 * 1024 * 1024, max_result_bytes: 8 * 1024 * 1024,
        aggregation: SentimentAggregationPolicy::default() }
}
fn compact() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 262_144,
        max_pair_checks: 1 << 25, max_pair_bytes: 1 << 28 }
}
fn total() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 32, max_output_tokens: 1 << 32,
        max_pair_checks: 1 << 36, max_pair_bytes: 1 << 40 }
}
fn make_planner(encoding: UntrustedEncoding) -> (SentimentPlanner, ExecutionIdentity) {
    let controls = pinned_controls::pinned().unwrap();
    let planner = SentimentPlanner::pinned(controls.template_controls(), SentimentOptions {
        mode: ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        eos_token_id: 166_101, policy: SentimentPolicy {
            minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
    }).unwrap().with_document_encoding(encoding).unwrap();
    let digest = Sha256Digest::of_bytes(b"sentiment document corpus contract fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: digest, artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(),
        packing_set_digest: digest, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "sentiment-v1".to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "none".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    (planner, identity)
}
fn fixture() -> (&'static SentimentPlanner, &'static ExecutionIdentity) {
    static FIXTURE: OnceLock<(SentimentPlanner, ExecutionIdentity)> = OnceLock::new();
    let (planner, identity) = FIXTURE.get_or_init(|| make_planner(UntrustedEncoding::ByteFallback));
    (planner, identity)
}
fn compact_fixture() -> (&'static SentimentPlanner, &'static ExecutionIdentity) {
    static FIXTURE: OnceLock<(SentimentPlanner, ExecutionIdentity)> = OnceLock::new();
    let (planner, identity) = FIXTURE.get_or_init(|| make_planner(UntrustedEncoding::SafeBpeV1(compact())));
    (planner, identity)
}
fn config(identity: &ExecutionIdentity, encoded: bool) -> Int8SentimentDocumentBatchConfig {
    Int8SentimentDocumentBatchConfig { identity: identity.clone(), task_ceiling: task(),
        planning: SentimentLimits::default(), mapping: mapping(),
        defaults: Some(SentimentBatchArgs { axes: vec![SentimentAxis::Valence], budget: task() }),
        max_model_work: work(), max_tokenizer_work: encoded.then(total) }
}
fn compiler(encoded: bool) -> Int8SentimentDocumentBatchPlanner<'static> {
    let (planner, identity) = if encoded { compact_fixture() } else { fixture() };
    Int8SentimentDocumentBatchPlanner::new(planner, config(identity, encoded)).unwrap()
}
fn ledger(compiler: &Int8SentimentDocumentBatchPlanner<'_>) -> RunState {
    RunState::new(compiler.config.max_model_work, compiler.config.max_tokenizer_work)
}
fn document(text: &str) -> BatchDocument<SentimentBatchArgs> {
    BatchDocument { id: "private-id".to_owned(), text: text.to_owned(), task_args: None }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: sequence * 64 }
}
struct RejectDriver {
    calls: usize,
    preflights: usize,
    reject_preflight: bool,
    cause: Option<DecodeCancellationKind>,
    clean: bool,
}
impl Default for RejectDriver {
    fn default() -> Self {
        Self { calls: 0, preflights: 0, reject_preflight: false, cause: None, clean: true }
    }
}
impl Driver for RejectDriver {
    fn clean(&self) -> bool { self.clean }
    fn capacity(&self) -> usize { 2048 }
    fn preflight<C: DecodeStepControl>(&mut self, _: &PreparedInt8SentimentMap<'_>,
        _: &[ExecutionIdentity], _: u64, _: &mut C) -> Result<(), Int8SentimentMapError> {
        self.preflights += 1;
        if self.reject_preflight { Err(Int8SentimentMapError::Work) } else { Ok(()) }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: &PreparedInt8SentimentMap<'_>,
        _: &[ExecutionIdentity], _: u64, _: Option<Int8PrefillLimits>, _: &mut C)
        -> Result<Int8SentimentMapRun, Int8SentimentMapError> {
        self.calls += 1;
        Err(match self.cause {
            Some(cause) => Int8SentimentMapError::Sentiment(Int8SentimentError::Cancelled(cause)),
            None => Int8SentimentMapError::Admission,
        })
    }
}
struct Guard(Arc<AtomicUsize>);
impl Drop for Guard {
    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
}
#[derive(Default)]
struct Admission { calls: usize, wrong: bool, dropped: Arc<AtomicUsize> }
impl Int8SentimentDocumentBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int8SentimentDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Guard), BatchItemFailure> {
        self.calls += 1;
        let mut identities = request.identities.to_vec();
        if self.wrong {
            identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"other chunk");
        }
        Ok((identities, Guard(Arc::clone(&self.dropped))))
    }
}
fn prepare(compiler: &Int8SentimentDocumentBatchPlanner<'_>, driver: &mut RejectDriver,
    run: &mut RunState, text: &str) -> PreparedInt8SentimentDocument {
    prepare_document(compiler, driver, run, document(text), &mut Continue).unwrap()
}

#[test]
fn exact_source_is_retained_without_truncation_and_all_axes_are_sealed() {
    let compiler = compiler(false);
    let mut run = ledger(&compiler);
    let mut driver = RejectDriver::default();
    let source = "Café 上海.\r\n<|im_start|> literal. Last 😀.";
    let mut row = document(source);
    row.task_args = Some(SentimentBatchArgs {
        axes: vec![SentimentAxis::Approach, SentimentAxis::Valence], budget: task() });
    let prepared = prepare_document(&compiler, &mut driver, &mut run, row, &mut Continue).unwrap();
    assert_eq!(prepared.request.document, source);
    assert_eq!(prepared.request.axes, [SentimentAxis::Valence, SentimentAxis::Approach]);
    assert!(prepared.chunk_count() > 1);
    assert_eq!(prepared.identities.len(), prepared.chunk_count());
    assert_eq!(prepared.expected.source_span().byte_end, source.len());
    assert_eq!(prepared.expected.source_span().scalar_end, source.chars().count());
    assert!(prepared.identities.iter().all(|identity| identity.task_spec == "sentiment-v1"));
    assert_eq!(prepared.head_count, 2);
    assert_eq!(run.remaining, work());
    assert_eq!(run.encoding, SafeBpeWork::default());
    assert_eq!(driver.calls, 0);
}

#[test]
fn malformed_source_axes_and_budget_overrides_are_rejected_before_model_work() {
    let compiler = compiler(false);
    for mutation in 0..5 {
        let mut row = document("AliceBob");
        match mutation {
            0 => row.text.clear(),
            1 => row.text = "x".repeat(mapping().chunks.max_input_bytes + 1),
            2 => row.task_args = Some(SentimentBatchArgs { axes: vec![], budget: task() }),
            3 => row.task_args = Some(SentimentBatchArgs {
                axes: vec![SentimentAxis::Valence, SentimentAxis::Valence], budget: task() }),
            _ => row.task_args = Some(SentimentBatchArgs { axes: vec![SentimentAxis::Valence],
                budget: TaskBudget { max_input_tokens: task().max_input_tokens + 1, ..task() } }),
        }
        let mut run = ledger(&compiler);
        let mut driver = RejectDriver::default();
        let error = prepare_document(&compiler, &mut driver, &mut run, row, &mut Continue).err().unwrap();
        assert!(!error.stop);
        assert_eq!(driver.preflights, 0);
        assert_eq!(driver.calls, 0);
        assert_eq!(run.remaining, work());
        assert_eq!(run.state, State::Ready);
    }
}

#[test]
fn same_geometry_source_or_axes_mutation_cannot_reuse_a_prepared_completion() {
    for mutation in 0..3 {
        let compiler = compiler(false);
        let mut run = ledger(&compiler);
        let mut driver = RejectDriver::default();
        let mut prepared = prepare(&compiler, &mut driver, &mut run, "AliceBob");
        match mutation {
            0 => prepared.request.document = "CarolDan".to_owned(),
            1 => prepared.request.axes = vec![SentimentAxis::Arousal],
            _ => prepared.request.budget.max_output_bytes -= 1,
        }
        let planned = prepared.planned_work();
        let mut admission = Admission::default();
        let error = execute_document(&compiler, &mut driver, &mut admission, &mut run, None,
            prepared, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop);
        assert_eq!(error.fault.code, BatchCode::Admission);
        assert_eq!(driver.calls, 0);
        assert_eq!(admission.calls, 0);
        assert_eq!(run.remaining, subtract(work(), planned).unwrap());
        assert_eq!(run.state, State::Failed);
    }
}

#[test]
fn identical_separate_factories_are_not_interchangeable_admission_authorities() {
    let first = compiler(false);
    let second = compiler(false);
    let mut first_run = ledger(&first);
    let mut second_run = ledger(&second);
    let mut driver = RejectDriver::default();
    let prepared = prepare(&first, &mut driver, &mut first_run, "AliceBob");
    let error = execute_document(&second, &mut driver, &mut Admission::default(), &mut second_run,
        None, prepared, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop);
    assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(driver.calls, 0);
    assert_eq!(second_run.remaining, work());
}

#[test]
fn last_chunk_identity_substitution_releases_the_guard_without_native_execution() {
    let compiler = compiler(false);
    let mut run = ledger(&compiler);
    let mut driver = RejectDriver::default();
    let prepared = prepare(&compiler, &mut driver, &mut run, "AliceBobCarolDan");
    assert!(prepared.identities.len() > 1);
    let planned = prepared.planned_work();
    let mut admission = Admission { wrong: true, ..Admission::default() };
    let error = execute_document(&compiler, &mut driver, &mut admission, &mut run, None,
        prepared, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop);
    assert_eq!(driver.calls, 0);
    assert_eq!(admission.calls, 1);
    assert_eq!(admission.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(run.remaining, subtract(work(), planned).unwrap());
    assert_eq!(run.state, State::Failed);
}

#[test]
fn every_native_attempt_cancellation_keeps_its_original_cause_and_full_debit() {
    use DecodeCancellationKind::*;
    let compiler = compiler(false);
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut run = ledger(&compiler);
        let mut driver = RejectDriver { cause: Some(cause), ..RejectDriver::default() };
        let prepared = prepare(&compiler, &mut driver, &mut run, "AliceBob");
        let planned = prepared.planned_work();
        let mut admission = Admission::default();
        let error = execute_document(&compiler, &mut driver, &mut admission, &mut run, None,
            prepared, context(1, 1), &mut Continue).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(cause));
        assert!(error.stop);
        assert_eq!(driver.calls, 1);
        assert_eq!(admission.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(run.remaining, subtract(work(), planned).unwrap());
        assert!(run.enter().is_err());
    }
}

#[test]
fn preparation_and_reconstruction_cancel_without_inventing_a_new_control() {
    let compiler = compiler(false);
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::User,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::ParentCancelled] {
        let mut run = ledger(&compiler);
        let mut driver = RejectDriver::default();
        let failure = prepare_document(&compiler, &mut driver, &mut run, document("AliceBob"),
            &mut Stop(cause)).err().unwrap();
        assert!(failure.stop);
        assert_eq!(failure.fault.cancellation, Some(cause));
        assert_eq!(driver.preflights, 0);
        assert_eq!(driver.calls, 0);

        let mut run = ledger(&compiler);
        let prepared = prepare(&compiler, &mut driver, &mut run, "AliceBob");
        let planned = prepared.planned_work();
        let failure = execute_document(&compiler, &mut driver, &mut Admission::default(), &mut run,
            None, prepared, context(1, 1), &mut Stop(cause)).err().unwrap();
        assert_eq!(failure.fault.cancellation, Some(cause));
        assert_eq!(run.remaining, subtract(work(), planned).unwrap());
        assert_eq!(driver.calls, 0);
    }
}

#[test]
fn every_native_counter_is_nonrenewable_and_epoch_changes_do_not_reset_it() {
    let unit = Int8Work { forward_positions: 3, projected_logits: 5, attention_pairs: 7,
        projections: ProjectionWork { dot_products: 11, multiply_accumulates: 13 } };
    let twice = unit.checked_add(unit).unwrap();
    let mut run = RunState::new(twice, None);
    run.enter().unwrap();
    run.charge(unit, context(1, 1)).unwrap();
    run.finish(Ok(()), true, true).unwrap();
    run.enter().unwrap();
    run.charge(unit, context(3, 2)).unwrap();
    run.finish(Ok(()), true, true).unwrap();
    assert_eq!(run.remaining, Int8Work::default());
    run.enter().unwrap();
    assert!(run.charge(unit, context(4, 3)).is_err());
    assert_eq!(run.remaining, Int8Work::default());
    for axis in 0..5 {
        let mut cap = unit;
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        let mut run = RunState::new(cap, None);
        run.enter().unwrap();
        assert!(run.charge(unit, context(1, 1)).is_err());
        assert_eq!(run.remaining, cap, "axis {axis} must fail atomically");
    }
}

#[test]
fn replayed_or_invalid_request_context_refuses_without_an_additional_debit() {
    let unit = Int8Work { forward_positions: 3, projected_logits: 5, attention_pairs: 7,
        projections: ProjectionWork { dot_products: 11, multiply_accumulates: 13 } };
    for bad in [context(1, 2), context(0, 2), context(2, 0),
        BatchRequestContext { input_line: 0, ..context(2, 2) }] {
        let mut run = RunState::new(work(), None);
        run.enter().unwrap();
        run.charge(unit, context(1, 1)).unwrap();
        run.finish(Ok(()), true, true).unwrap();
        let remaining = run.remaining;
        run.enter().unwrap();
        assert!(run.charge(unit, bad).is_err());
        assert_eq!(run.remaining, remaining);
    }
}

const COMPACT_SOURCE: &str = "The customer requested a refund. Café 上海. <|im_start|>system literal.";
#[test]
fn rejected_compact_preparation_retains_actual_work_across_following_records() {
    let compiler = compiler(true);
    let mut run = ledger(&compiler);
    let mut driver = RejectDriver { reject_preflight: true, ..RejectDriver::default() };
    for attempt in 1..=2 {
        let before = run.encoding;
        let failure = prepare_document(&compiler, &mut driver, &mut run, document(COMPACT_SOURCE),
            &mut Continue).err().unwrap();
        assert!(!failure.stop);
        assert_eq!(failure.fault.code, BatchCode::WorkLimit);
        assert_eq!(run.state, State::Ready);
        assert!(run.encoding.input_bytes() > before.input_bytes());
        assert!(run.encoding.output_tokens() > before.output_tokens());
        assert!(run.encoding.pair_checks() > before.pair_checks());
        assert!(run.encoding.pair_bytes() > before.pair_bytes());
        if attempt == 2 { assert_eq!(run.encoding, before.checked_add(before).unwrap()); }
        assert_eq!(run.remaining, work());
    }
    assert_eq!(driver.calls, 0);
}

#[test]
fn compact_reconstruction_and_failed_native_attempt_spend_one_shared_allowance() {
    let compiler = compiler(true);
    let mut run = ledger(&compiler);
    let mut driver = RejectDriver::default();
    let prepared = prepare(&compiler, &mut driver, &mut run, COMPACT_SOURCE);
    let first = prepared.encoding_work();
    assert_eq!(run.encoding, first);
    let failure = execute_document(&compiler, &mut driver, &mut Admission::default(), &mut run,
        None, prepared, context(1, 1), &mut Continue).err().unwrap();
    // The private driver intentionally rejects after exact reconstruction.
    assert_eq!(driver.calls, 1);
    assert!(failure.stop);
    assert_eq!(run.encoding, first.checked_add(first).unwrap());
}

fn exact_total(work: SafeBpeWork) -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: work.input_bytes() as u64,
        max_output_tokens: work.output_tokens() as u64, max_pair_checks: work.pair_checks(),
        max_pair_bytes: work.pair_bytes() }
}
#[test]
fn a_tight_execution_allowance_preserves_original_policy_identity_and_exact_plan() {
    let compiler = compiler(true);
    let mut ordinary = ledger(&compiler);
    let mut driver = RejectDriver::default();
    let expected = prepare(&compiler, &mut driver, &mut ordinary, COMPACT_SOURCE);
    let mut tight = RunState::new(work(), Some(exact_total(expected.encoding_work())));
    let actual = prepare(&compiler, &mut driver, &mut tight, COMPACT_SOURCE);
    assert_eq!(actual.expected, expected.expected);
    assert_eq!(actual.identities, expected.identities);
    assert_eq!(actual.encoding_work(), expected.encoding_work());
    assert_eq!(compiler.document_encoding(), &UntrustedEncoding::SafeBpeV1(compact()));
    let failed = prepare_document(&compiler, &mut driver, &mut tight, document(COMPACT_SOURCE),
        &mut Continue).err().unwrap();
    assert!(failed.stop);
    assert_eq!(failed.fault.code, BatchCode::WorkLimit);
    assert_eq!(tight.encoding, expected.encoding_work());
}

#[test]
fn exhausted_corpus_encoding_refuses_reconstruction_before_admission_or_native_work() {
    let compiler = compiler(true);
    let mut baseline = ledger(&compiler);
    let mut driver = RejectDriver::default();
    let expected = prepare(&compiler, &mut driver, &mut baseline, COMPACT_SOURCE);
    let mut run = RunState::new(work(), Some(exact_total(expected.encoding_work())));
    let prepared = prepare(&compiler, &mut driver, &mut run, COMPACT_SOURCE);
    let planned = prepared.planned_work();
    let mut admission = Admission::default();
    let error = execute_document(&compiler, &mut driver, &mut admission, &mut run,
        None, prepared, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop);
    assert_eq!(error.fault.code, BatchCode::WorkLimit);
    assert_eq!(driver.calls, 0);
    assert_eq!(admission.calls, 0);
    assert_eq!(run.remaining, subtract(work(), planned).unwrap());
    assert_eq!(run.encoding, expected.encoding_work());
}

#[test]
fn each_cumulative_encoding_axis_can_refuse_before_its_requested_work_occurs() {
    let compiler = compiler(true);
    let mut baseline = ledger(&compiler);
    let mut driver = RejectDriver::default();
    let expected = prepare(&compiler, &mut driver, &mut baseline, COMPACT_SOURCE);
    for axis in 0..4 {
        let mut limits = exact_total(expected.encoding_work());
        match axis { 0 => limits.max_input_bytes -= 1, 1 => limits.max_output_tokens -= 1,
            2 => limits.max_pair_checks -= 1, _ => limits.max_pair_bytes -= 1 }
        let mut run = RunState::new(work(), Some(limits));
        let failure = prepare_document(&compiler, &mut driver, &mut run, document(COMPACT_SOURCE),
            &mut Continue).err().unwrap();
        assert!(failure.stop);
        assert_eq!(failure.fault.code, BatchCode::WorkLimit);
        assert!(encoding_fits(run.encoding, limits));
        assert_eq!(driver.calls, 0);
    }
}

#[test]
fn invalid_fixed_configuration_cannot_be_deferred_until_after_model_load() {
    let (planner, identity) = fixture();
    for axis in 0..5 {
        let mut invalid = config(identity, false);
        match axis { 0 => invalid.identity.task_spec = "classify-v1".to_owned(),
            1 => invalid.identity.template_digest = Sha256Digest::of_bytes(b"other policy"),
            2 => invalid.max_model_work.forward_positions = 0,
            3 => invalid.planning.max_total_prompt_tokens = 0,
            _ => invalid.max_tokenizer_work = Some(total()) }
        assert!(invalid.validate(planner).is_err());
    }
    let (planner, identity) = compact_fixture();
    let mut invalid = config(identity, true);
    invalid.max_tokenizer_work = None;
    assert!(invalid.validate(planner).is_err());
}

#[test]
fn full_declared_cache_is_priced_for_all_axes_even_if_selected_documents_are_short() {
    assert!(capacity_bytes(2048, task(), 4).is_ok());
    let smaller = TaskBudget { max_kv_bytes: task().max_kv_bytes - 1, ..task() };
    assert!(capacity_bytes(2048, smaller, 4).is_err());
    assert!(capacity_bytes(0, task(), 1).is_err());
    assert!(capacity_bytes(2048, task(), 0).is_err());
    assert!(capacity_bytes(2048, task(), 5).is_err());
}

#[test]
fn identity_metadata_bound_counts_owned_capacity_before_chunk_cloning() {
    let (planner, identity) = fixture();
    config(identity, false).validate(planner).unwrap();
    for overallocated in [false, true] {
        let mut invalid = config(identity, false);
        let mut text = String::with_capacity(MAX_IDENTITY_TEXT_BYTES as usize + 1);
        text.push_str(if overallocated { "short" } else { "" });
        if !overallocated { text.push_str(&"x".repeat(MAX_IDENTITY_TEXT_BYTES as usize + 1)); }
        invalid.identity.grammar_compiler_version = text;
        let error = invalid.validate(planner).unwrap_err();
        assert_eq!(error.code, BatchCode::InvalidLimits);
    }
}

#[test]
fn default_axis_storage_prices_moved_spare_capacity_and_join_keepalive() {
    let (_, identity) = fixture();
    let mut config = config(identity, false);
    let mut axes = Vec::with_capacity(1024);
    axes.push(SentimentAxis::Valence);
    let owned = axes.capacity();
    config.defaults = Some(SentimentBatchArgs { axes, budget: task() });
    assert_eq!(config.retained_defaults_bytes().unwrap(),
        (owned + 1) as u64 * std::mem::size_of::<SentimentAxis>() as u64);
    config.defaults = None;
    assert_eq!(config.retained_defaults_bytes().unwrap(), 0);
}
