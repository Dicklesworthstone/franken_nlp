//! Real pinned planning and model-free ownership/accounting regressions.
//! These definitions do not establish native model quality or execution parity.
use super::*;
use std::sync::OnceLock;
use crate::{
    execution_identity::Sha256Digest,
    native_engine::decode::DecodeCancellationKind,
    tasks::{judge::FaithfulnessAggregationPolicy, mapreduce::ChunkLimits},
    tokenizer::{pinned_controls, embedded::EmbeddedTokenizer, untrusted::SafeBpeLimits},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn planner(compact: bool) -> Arc<JudgePlanner> {
    static BYTE: OnceLock<Arc<JudgePlanner>> = OnceLock::new();
    static COMPACT: OnceLock<Arc<JudgePlanner>> = OnceLock::new();
    let cell = if compact { &COMPACT } else { &BYTE };
    Arc::clone(cell.get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let planner = JudgePlanner::pinned(controls.template_controls(), eos).unwrap();
        Arc::new(if compact { planner.with_document_encoding(UntrustedEncoding::SafeBpeV1(call())).unwrap() }
            else { planner })
    }))
}
fn call() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 262_144,
        max_pair_checks: 1 << 25, max_pair_bytes: 1 << 28 }
}
fn tokenizer_cap() -> SafeBpeWorkLimits {
    let call = call();
    SafeBpeWorkLimits { max_input_bytes: call.max_input_bytes as u64 * 8,
        max_output_tokens: call.max_output_tokens as u64 * 8,
        max_pair_checks: call.max_pair_checks * 8, max_pair_bytes: call.max_pair_bytes * 8 }
}
fn work() -> Int8Work {
    Int8Work { forward_positions: 1_000_000, projected_logits: 1_000_000_000_000,
        attention_pairs: 1_000_000_000_000,
        projections: ProjectionWork { dot_products: 1_000_000_000_000,
            multiply_accumulates: 1_000_000_000_000_000_000 } }
}
fn identity(planner: &JudgePlanner) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"synthetic-int8-faithfulness-corpus");
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1, source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: absent, artifact_format: "synthetic-only".to_owned(),
        quant_recipe: "fixture-int8".to_owned(), packing_set_digest: absent,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "judge-v1".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }).unwrap()
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 8, max_output_bytes: 1_000_000,
        max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(1024).unwrap().kv_bytes * 32 }
}
fn policy() -> FaithfulnessPolicy {
    FaithfulnessPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_milli: 100,
        evidence_window_bytes: 8, max_evidence_windows: 31, max_evidence_spans: 31 }
}
fn mapping() -> Int8FaithfulnessMapLimits {
    Int8FaithfulnessMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 16, max_chunk_tokens: 1024,
            context_tokens: 1024, reserved_tokens: 1, max_chunks: 64, max_tokenizer_calls: 4096 },
        max_model_work: work(), max_retained_result_bytes: 4 << 20, max_result_bytes: 8 << 20,
        aggregation: FaithfulnessAggregationPolicy::default(),
    }
}
fn compiler(compact: bool) -> Int8FaithfulnessDocumentBatchPlanner {
    let planner = planner(compact); let identity = identity(&planner);
    Int8FaithfulnessDocumentBatchPlanner::new(planner, identity, task(), JudgeLimits::default(),
        mapping(), policy(), work(), compact.then(tokenizer_cap)).unwrap()
}
fn document() -> BatchDocument<Int8FaithfulnessDocumentArgs> {
    BatchDocument { id: "caller-label".to_owned(),
        text: "é\r\n𐐀 one <think> literal. Second chunk.\nTail.".to_owned(),
        task_args: Some(Int8FaithfulnessDocumentArgs {
            claim: "The claim <|im_start|> remains data.".to_owned() }) }
}
fn ledger(compiler: &Int8FaithfulnessDocumentBatchPlanner) -> EncodingLedger {
    EncodingLedger::new(compiler.document_encoding(), compiler.max_tokenizer_work()).unwrap()
}
fn prepared(compiler: &Int8FaithfulnessDocumentBatchPlanner, document: BatchDocument<Int8FaithfulnessDocumentArgs>,
    ledger: &mut EncodingLedger, invocation: &Arc<()>) -> Result<PreparedInt8FaithfulnessDocument, BatchItemFailure> {
    compiler.prepare(document, compiler.max_run_work(), ledger, invocation, &mut Continue, |_, _, _| Ok(()))
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}

#[test]
fn every_original_source_and_claim_match_the_direct_complete_document_plan() {
    for compact in [false, true] {
        let compiler = compiler(compact); let invocation = Arc::new(());
        let mut ledger = ledger(&compiler); let document = document();
        let source = document.text.clone(); let claim = document.task_args.as_ref().unwrap().claim.clone();
        let item = prepared(&compiler, document, &mut ledger, &invocation).unwrap();
        let JudgeRequest::Faithfulness { source: exact_source, claim: exact_claim, policy, budget } = &item.request
            else { panic!("complete-source faithfulness only"); };
        assert_eq!(exact_source, &source); assert_eq!(exact_claim, &claim);
        assert_eq!(*policy, compiler.policy); assert_eq!(*budget, compiler.ceiling);
        assert!(item.chunk_count() > 1);
        let direct = compiler.planner.plan_int8_faithfulness_document_with_control(&item.request,
            &PlanContext::new(&compiler.identity, compiler.ceiling).unwrap(),
            compiler.planning, compiler.mapping, &mut Continue).unwrap();
        assert!(direct.max_head_count() > 1);
        assert_eq!(item.expected, direct.preflight_metadata());
        assert_eq!(item.execution_identities(), direct.identities().unwrap());
        assert_eq!(item.planned_work(), direct.planned_work());
        assert_eq!(ledger.used, item.encoding_work());
        compiler.verify(&item, &invocation).unwrap();
    }
}

#[test]
fn reconstruction_preserves_native_binding_and_charges_both_real_planning_passes() {
    let compiler = compiler(true); let mut ledger = ledger(&compiler); let invocation = Arc::new(());
    let mut item = prepared(&compiler, document(), &mut ledger, &invocation).unwrap();
    let first = item.encoding_work;
    assert!(first.input_bytes() > document().text.len());
    let plan = compiler.plan(&item.request, &mut item.encoding_work, &mut ledger, &mut Continue, true).unwrap();
    assert_eq!(plan.preflight_metadata(), item.expected);
    assert_eq!(plan.identities().unwrap(), item.identities);
    assert_eq!(item.encoding_work, first.checked_add(first).unwrap());
    assert_eq!(ledger.used, item.encoding_work);
}

#[test]
fn late_partition_rejection_retains_spent_compact_work_before_the_next_record() {
    let mut compiler = compiler(true); compiler.mapping.chunks.max_chunks = 1;
    let mut ledger = ledger(&compiler); let invocation = Arc::new(());
    let failure = prepared(&compiler, document(), &mut ledger, &invocation).err().unwrap();
    assert!(!failure.stop);
    let spent = ledger.used; assert!(spent.input_bytes() > 0);
    let mut next = document(); next.text = "é".to_owned();
    let item = prepared(&compiler, next, &mut ledger, &invocation).unwrap();
    assert_eq!(item.chunk_count(), 1);
    assert_eq!(ledger.used, spent.checked_add(item.encoding_work).unwrap());
}

#[test]
fn each_invocation_tokenizer_axis_refuses_without_renewing_at_the_next_record() {
    let compiler = compiler(true); let invocation = Arc::new(());
    let item = prepared(&compiler, document(), &mut ledger(&compiler), &invocation).unwrap();
    let used = item.encoding_work; let call = call();
    assert!(used.input_bytes() > 0 && used.output_tokens() > 0 && used.pair_checks() > 0 && used.pair_bytes() > 0);
    for axis in 0..4 {
        let mut total = tokenizer_cap();
        match axis {
            0 => total.max_input_bytes = call.max_input_bytes as u64 + used.input_bytes() as u64 - 1,
            1 => total.max_output_tokens = call.max_output_tokens as u64 + used.output_tokens() as u64 - 1,
            2 => total.max_pair_checks = call.max_pair_checks + used.pair_checks() - 1,
            _ => total.max_pair_bytes = call.max_pair_bytes + used.pair_bytes() - 1,
        }
        let mut ledger = EncodingLedger::new(compiler.document_encoding(), Some(total)).unwrap();
        prepared(&compiler, document(), &mut ledger, &invocation).unwrap();
        let failure = prepared(&compiler, document(), &mut ledger, &invocation).err().unwrap();
        assert_eq!(failure.fault.code, BatchCode::WorkLimit); assert!(failure.stop);
        assert_eq!(ledger.used, used, "axis {axis} performed work after exhausted admission");
    }
}

#[test]
fn execution_reconstruction_cannot_acquire_a_fresh_document_encoding_allowance() {
    let first = compiler(true); let invocation = Arc::new(());
    let item = prepared(&first, document(), &mut ledger(&first), &invocation).unwrap();
    let actual = item.encoding_work;
    let exact = SafeBpeLimits { max_input_bytes: actual.input_bytes(), max_output_tokens: actual.output_tokens(),
        max_pair_checks: actual.pair_checks(), max_pair_bytes: actual.pair_bytes() };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = Arc::new(JudgePlanner::pinned(controls.template_controls(), eos).unwrap()
        .with_document_encoding(UntrustedEncoding::SafeBpeV1(exact)).unwrap());
    let identity = identity(&planner);
    let compiler = Int8FaithfulnessDocumentBatchPlanner::new(planner, identity, task(), JudgeLimits::default(),
        mapping(), policy(), work(), Some(tokenizer_cap())).unwrap();
    let mut ledger = ledger(&compiler);
    let mut item = prepared(&compiler, document(), &mut ledger, &invocation).unwrap();
    let spent = ledger.used;
    let failure = compiler.plan(&item.request, &mut item.encoding_work, &mut ledger,
        &mut Continue, true).err().unwrap();
    assert!(failure.stop); assert_eq!(ledger.used, spent); assert_eq!(item.encoding_work, spent);
}

#[test]
fn cancelled_preparation_keeps_original_cause_and_actual_tokenizer_debt() {
    struct Control { calls: usize, stop: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == self.stop).then_some(self.cause)
        }
    }
    let compiler = compiler(true); let invocation = Arc::new(());
    let mut baseline = Control { calls: 0, stop: usize::MAX, cause: DecodeCancellationKind::User };
    compiler.prepare(document(), work(), &mut ledger(&compiler), &invocation,
        &mut baseline, |_, _, _| Ok(())).unwrap();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::LinkedExit] {
        let mut control = Control { calls: 0, stop: baseline.calls - 1, cause };
        let mut ledger = ledger(&compiler);
        let failure = compiler.prepare(document(), work(), &mut ledger, &invocation,
            &mut control, |_, _, _| Ok(())).err().unwrap();
        assert_eq!(failure.fault.cancellation, Some(cause)); assert!(failure.stop);
        assert!(ledger.used.input_bytes() > 0);
    }
}

#[test]
fn identical_public_configuration_cannot_substitute_prepared_factory_or_invocation() {
    let other = compiler(false); let compiler = compiler(false);
    let invocation = Arc::new(()); let other_invocation = Arc::new(());
    let item = prepared(&compiler, document(), &mut ledger(&compiler), &invocation).unwrap();
    compiler.verify(&item, &invocation).unwrap();
    assert_eq!(compiler.verify(&item, &other_invocation).unwrap_err().fault.code, BatchCode::Admission);
    assert_eq!(other.verify(&item, &invocation).unwrap_err().fault.code, BatchCode::Admission);
}

#[test]
fn strict_records_accept_only_claim_and_cannot_override_fixed_policy_or_task_budget() {
    let valid: BatchDocument<Int8FaithfulnessDocumentArgs> = serde_json::from_str(
        r#"{"id":"row","text":"exact source","task_args":{"claim":"exact claim"}}"#).unwrap();
    let compiler = compiler(false);
    let request = compiler.request(valid).unwrap();
    let JudgeRequest::Faithfulness { source, claim, policy, budget } = request
        else { panic!("faithfulness record"); };
    assert_eq!(source, "exact source"); assert_eq!(claim, "exact claim");
    assert_eq!(policy, compiler.policy); assert_eq!(budget, compiler.ceiling);
    for json in [
        r#"{"claim":"claim","source":"alternate source"}"#,
        r#"{"claim":"claim","policy":{}}"#,
        r#"{"claim":"claim","budget":{}}"#,
        r#"{}"#,
    ] { assert!(serde_json::from_str::<Int8FaithfulnessDocumentArgs>(json).is_err()); }
    for axis in 0..5 {
        let mut record = document();
        match axis {
            0 => record.task_args = None,
            1 => record.text.clear(),
            2 => record.task_args.as_mut().unwrap().claim.clear(),
            3 => record.task_args.as_mut().unwrap().claim = "x".repeat(mapping().chunks.max_input_bytes),
            _ => record.text.reserve(mapping().chunks.max_input_bytes * 8),
        }
        assert!(compiler.request(record).is_err(), "axis {axis}");
    }
}

#[test]
fn malformed_fixed_policy_identity_and_encoding_envelopes_refuse_before_private_input() {
    let planner = planner(false); let identity = identity(&planner);
    let mut invalid = policy(); invalid.max_evidence_windows = 32;
    assert!(Int8FaithfulnessDocumentBatchPlanner::validate_configuration(&planner, &identity,
        task(), JudgeLimits::default(), mapping(), invalid, work(), None).is_err());
    assert!(Int8FaithfulnessDocumentBatchPlanner::validate_configuration(&planner, &identity,
        task(), JudgeLimits::default(), mapping(), policy(), work(), Some(tokenizer_cap())).is_err());
    let mut altered = identity.clone(); altered.backend_semantic_version = "untrusted-backend".to_owned();
    assert!(Int8FaithfulnessDocumentBatchPlanner::validate_configuration(&planner, &altered,
        task(), JudgeLimits::default(), mapping(), policy(), work(), None).is_err());
    let compact = compiler(true);
    assert!(Int8FaithfulnessDocumentBatchPlanner::validate_configuration(&compact.planner, &compact.identity,
        task(), JudgeLimits::default(), mapping(), policy(), work(), None).is_err());
}

#[test]
fn all_five_native_debits_are_atomic_nonrefundable_and_survive_flush_epochs() {
    let mut run = RunState::new(work());
    let debit = Int8Work { forward_positions: 5, projected_logits: 6, attention_pairs: 7,
        projections: ProjectionWork { dot_products: 8, multiply_accumulates: 9 } };
    run.enter().unwrap(); run.charge(debit, context(1, 1)).unwrap();
    run.finish(Ok(()), true, true).unwrap();
    run.enter().unwrap(); run.charge(debit, context(2, 2)).unwrap();
    assert_eq!(run.remaining, subtract(subtract(work(), debit).unwrap(), debit).unwrap());
    let remaining = run.remaining;
    assert!(run.finish::<()>(Err(BatchItemFailure::reject(BatchCode::Admission)), true, true).unwrap_err().stop);
    assert_eq!(run.remaining, remaining); assert!(run.enter().is_err());
    for axis in 0..5 {
        let mut short = debit;
        match axis { 0 => short.forward_positions -= 1, 1 => short.projected_logits -= 1,
            2 => short.attention_pairs -= 1, 3 => short.projections.dot_products -= 1,
            _ => short.projections.multiply_accumulates -= 1 }
        let mut run = RunState::new(short); run.enter().unwrap();
        assert_eq!(run.charge(debit, context(1, 1)).unwrap_err().fault.code, BatchCode::WorkLimit);
        assert_eq!(run.remaining, short); assert_eq!(run.last_sequence, 0);
    }
}

#[test]
fn cancellation_unclean_exit_and_unwind_each_prevent_reusing_a_stream_owner() {
    for axis in 0..3 {
        let mut run = RunState::new(work()); run.enter().unwrap();
        match axis {
            0 => {
                let error = BatchItemFailure::reject(BatchFault::cancelled(DecodeCancellationKind::Deadline));
                assert!(run.finish::<()>(Err(error), true, false).unwrap_err().stop);
            }
            1 => { assert!(run.finish(Ok(()), false, false).unwrap_err().stop); }
            _ => {
                let result = std::panic::catch_unwind(|| panic!("synthetic interrupted callback"));
                assert!(result.is_err());
            }
        }
        assert!(run.enter().is_err(), "axis {axis}");
    }
}

#[test]
fn repeated_or_invalid_transport_sequence_cannot_reexecute_native_work() {
    for invalid in [context(0, 1), context(1, 0), BatchRequestContext {
        request_seq: 1, epoch: 1, input_line: 0, byte_offset: 0 }] {
        let mut run = RunState::new(work()); run.enter().unwrap();
        assert_eq!(run.charge(work(), invalid).unwrap_err().fault.code, BatchCode::InvalidExecution);
        assert_eq!(run.remaining, work());
    }
    let mut run = RunState::new(work()); run.enter().unwrap();
    run.charge(Int8Work::default(), context(1, 1)).unwrap();
    run.finish(Ok(()), true, true).unwrap(); run.enter().unwrap();
    assert_eq!(run.charge(Int8Work::default(), context(1, 2)).unwrap_err().fault.code, BatchCode::InvalidExecution);
    assert_eq!(run.remaining, work());
}
