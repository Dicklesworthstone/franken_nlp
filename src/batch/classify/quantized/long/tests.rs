//! Real pinned planning plus model-free ownership and accounting regressions.
//! Native model equality is a separate explicitly armed validation gate.
use super::*;
use std::sync::OnceLock;
use crate::{
    tasks::{classify::{ClassificationAggregationPolicy, ClassificationPolicy},
        mapreduce::ChunkLimits},
    tokenizer::{pinned_controls, embedded::EmbeddedTokenizer, untrusted::SafeBpeLimits},
};

fn planner(compact: bool) -> Arc<ClassificationPlanner> {
    static BYTE: OnceLock<Arc<ClassificationPlanner>> = OnceLock::new();
    static COMPACT: OnceLock<Arc<ClassificationPlanner>> = OnceLock::new();
    let cell = if compact { &COMPACT } else { &BYTE };
    Arc::clone(cell.get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let planner = ClassificationPlanner::pinned(controls.template_controls(), eos).unwrap();
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
fn identity(planner: &ClassificationPlanner) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1, source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: absent, artifact_format: "synthetic-only".to_owned(),
        quant_recipe: "fixture-int8".to_owned(), packing_set_digest: absent,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "classify-v1".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }).unwrap()
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(1024).unwrap().kv_bytes }
}
fn planning() -> ClassificationLimits {
    ClassificationLimits { max_labels: 4, max_input_bytes: 4096,
        max_total_label_bytes: 1024, max_context_tokens: 1024, ..Default::default() }
}
fn mapping() -> Int8ClassificationMapLimits {
    Int8ClassificationMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8,
            max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16,
            max_chunks: 64, max_tokenizer_calls: 4096 },
        max_model_work: work(), max_retained_result_bytes: 1 << 20, max_result_bytes: 1 << 20,
        aggregation: ClassificationAggregationPolicy::default(),
    }
}
fn args() -> ClassificationBatchArgs {
    ClassificationBatchArgs {
        labels: ["alpha", "beta"].map(|id| ClassificationLabel {
            id: id.to_owned(), description: format!("Private definition of {id}") }).to_vec(),
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: task(),
    }
}
fn compiler(compact: bool) -> Int8ClassificationDocumentBatchPlanner {
    let planner = planner(compact); let identity = identity(&planner);
    Int8ClassificationDocumentBatchPlanner::new(planner, identity, task(), planning(),
        mapping(), Some(args()), work(), compact.then(tokenizer_cap)).unwrap()
}
fn document() -> BatchDocument<ClassificationBatchArgs> {
    BatchDocument { id: "caller-label".to_owned(), text: "é\r\n𐐀 <think> all original source bytes".to_owned(),
        task_args: None }
}
fn ledger(compiler: &Int8ClassificationDocumentBatchPlanner) -> EncodingLedger {
    EncodingLedger::new(compiler.document_encoding(), compiler.max_tokenizer_work()).unwrap()
}
fn prepared(compiler: &Int8ClassificationDocumentBatchPlanner, document: BatchDocument<ClassificationBatchArgs>,
    ledger: &mut EncodingLedger, invocation: &Arc<()>)
    -> Result<PreparedInt8ClassificationDocument, BatchItemFailure> {
    compiler.prepare(document, compiler.max_run_work(), ledger, invocation,
        &mut Continue, |_, _, _| Ok(()))
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}

#[test]
fn both_classification_modes_prepare_every_original_chunk_and_match_the_one_document_plan() {
    for compact in [false, true] {
        for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
            let compiler = compiler(compact); let invocation = Arc::new(());
            let mut ledger = ledger(&compiler); let mut document = document();
            let mut settings = args(); settings.mode = mode; document.task_args = Some(settings);
            let expected_source = document.text.clone();
            let item = prepared(&compiler, document, &mut ledger, &invocation).unwrap();
            assert_eq!(item.request.document, expected_source);
            assert!(item.chunk_count() > 1);
            let direct = compiler.planner.plan_int8_document_with_control(&item.request,
                &PlanContext::new(&compiler.identity, compiler.ceiling).unwrap(),
                compiler.planning, compiler.mapping, &mut Continue).unwrap();
            assert_eq!(item.expected, direct.preflight_metadata());
            assert_eq!(item.execution_identities(), direct.identities().unwrap());
            assert_eq!(item.planned_work(), direct.planned_work());
            assert_eq!(direct.head_count(), if mode == ClassificationMode::Exclusive { 1 } else { 2 });
            assert_eq!(ledger.used, item.encoding_work());
            compiler.verify(&item, &invocation).unwrap();
        }
    }
}

#[test]
fn reconstruction_keeps_exact_identity_and_charges_both_real_planning_passes() {
    let compiler = compiler(true); let mut ledger = ledger(&compiler); let invocation = Arc::new(());
    let mut item = prepared(&compiler, document(), &mut ledger, &invocation).unwrap();
    let first = item.encoding_work;
    assert!(first.input_bytes() > item.request.document.len());
    let plan = compiler.plan(&item.request, &mut item.encoding_work, &mut ledger, &mut Continue, true).unwrap();
    assert_eq!(plan.preflight_metadata(), item.expected);
    assert_eq!(plan.identities().unwrap(), item.identities);
    assert_eq!(item.encoding_work, first.checked_add(first).unwrap());
    assert_eq!(ledger.used, item.encoding_work);
}

#[test]
fn failed_late_partition_keeps_spent_compact_work_before_the_next_document() {
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
fn all_four_run_encoding_caps_refuse_before_another_preparation_can_start() {
    let compiler = compiler(true); let invocation = Arc::new(());
    let mut baseline = ledger(&compiler);
    let item = prepared(&compiler, document(), &mut baseline, &invocation).unwrap();
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
        assert_eq!(ledger.used, used);
        let failure = prepared(&compiler, document(), &mut ledger, &invocation).err().unwrap();
        assert_eq!(failure.fault.code, BatchCode::WorkLimit); assert!(failure.stop);
        assert_eq!(ledger.used, used, "axis {axis} performed work after exhausted admission");
    }
}

#[test]
fn reconstruction_does_not_renew_the_original_document_encoding_allowance() {
    let first = compiler(true); let invocation = Arc::new(());
    let item = prepared(&first, document(), &mut ledger(&first), &invocation).unwrap();
    let actual = item.encoding_work;
    let exact = SafeBpeLimits { max_input_bytes: actual.input_bytes(), max_output_tokens: actual.output_tokens(),
        max_pair_checks: actual.pair_checks(), max_pair_bytes: actual.pair_bytes() };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = Arc::new(ClassificationPlanner::pinned(controls.template_controls(), eos).unwrap()
        .with_document_encoding(UntrustedEncoding::SafeBpeV1(exact)).unwrap());
    let identity = identity(&planner);
    let compiler = Int8ClassificationDocumentBatchPlanner::new(planner, identity, task(), planning(),
        mapping(), Some(args()), work(), Some(tokenizer_cap())).unwrap();
    let mut ledger = ledger(&compiler);
    let mut item = prepared(&compiler, document(), &mut ledger, &invocation).unwrap();
    let spent = ledger.used;
    let failure = compiler.plan(&item.request, &mut item.encoding_work, &mut ledger,
        &mut Continue, true).err().unwrap();
    assert!(failure.stop);
    assert!(ledger.used.input_bytes() >= spent.input_bytes());
    assert_eq!(item.encoding_work, spent);
}

#[test]
fn failed_preparation_preserves_each_original_cancellation_cause_and_work_debt() {
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
fn identical_configuration_cannot_substitute_another_factory_or_invocation() {
    let other = compiler(false); let compiler = compiler(false);
    let invocation = Arc::new(()); let other_invocation = Arc::new(());
    let item = prepared(&compiler, document(), &mut ledger(&compiler), &invocation).unwrap();
    compiler.verify(&item, &invocation).unwrap();
    assert_eq!(compiler.verify(&item, &other_invocation).unwrap_err().fault.code, BatchCode::Admission);
    assert_eq!(other.verify(&item, &invocation).unwrap_err().fault.code, BatchCode::Admission);
}

#[test]
fn per_record_arguments_replace_defaults_and_only_shrink_the_task_envelope() {
    let compiler = compiler(false); let mut replacement = args();
    replacement.labels.truncate(1); replacement.labels[0].id = "new-label".to_owned();
    replacement.mode = ClassificationMode::MultiLabel; replacement.budget.max_output_bytes /= 2;
    let mut record = document(); record.task_args = Some(replacement.clone());
    let request = compiler.request(record).unwrap();
    assert!(request.labels == replacement.labels); assert_eq!(request.mode, replacement.mode);
    assert_eq!(request.budget, replacement.budget);
    for axis in 0..5 {
        let mut oversized = args();
        match axis { 0 => oversized.budget.max_input_tokens += 1, 1 => oversized.budget.max_output_tokens += 1,
            2 => oversized.budget.max_output_bytes += 1, 3 => oversized.budget.max_grammar_states += 1,
            _ => oversized.budget.max_kv_bytes += 1 }
        let mut document = document(); document.task_args = Some(oversized);
        assert_eq!(compiler.request(document).err().unwrap().fault.code, BatchCode::Planning);
    }
}

#[test]
fn malformed_fixed_defaults_and_encoding_envelopes_fail_before_population_preparation() {
    let base_planner = planner(false); let base_identity = identity(&base_planner);
    let mut duplicate = args(); duplicate.labels[1].id = duplicate.labels[0].id.clone();
    assert!(Int8ClassificationDocumentBatchPlanner::validate_configuration(&base_planner, &base_identity,
        task(), planning(), mapping(), Some(&duplicate), work(), None).is_err());
    assert!(Int8ClassificationDocumentBatchPlanner::validate_configuration(&base_planner, &base_identity,
        task(), planning(), mapping(), Some(&args()), work(), Some(tokenizer_cap())).is_err());
    let compact = planner(true); let compact_identity = identity(&compact);
    assert!(Int8ClassificationDocumentBatchPlanner::validate_configuration(&compact, &compact_identity,
        task(), planning(), mapping(), Some(&args()), work(), None).is_err());
}

#[test]
fn owned_default_capacity_is_priced_even_when_its_logical_string_is_short() {
    let mut compiler = compiler(false);
    let before = compiler.retained_defaults_bytes().unwrap();
    let old_capacity = compiler.defaults.as_ref().unwrap().labels[0].description.capacity() as u64;
    compiler.defaults.as_mut().unwrap().labels[0].description.reserve(1024 * 1024);
    let new_capacity = compiler.defaults.as_ref().unwrap().labels[0].description.capacity() as u64;
    assert_eq!(compiler.retained_defaults_bytes().unwrap(), before - old_capacity + new_capacity);
    assert!(new_capacity >= 1024 * 1024 && new_capacity > old_capacity);
}

#[test]
fn all_five_native_debits_are_atomic_nonrefundable_and_survive_flush_epochs() {
    let mut run = DocumentRunState::new(work());
    let debit = Int8Work { forward_positions: 5, projected_logits: 6, attention_pairs: 7,
        projections: ProjectionWork { dot_products: 8, multiply_accumulates: 9 } };
    run.enter().unwrap(); run.charge(debit, context(1, 1)).unwrap();
    run.finish(Ok(()), true, true).unwrap();
    run.enter().unwrap(); run.charge(debit, context(2, 2)).unwrap();
    assert_eq!(run.remaining, subtract(subtract(work(), debit).unwrap(), debit).unwrap());
    let remaining = run.remaining;
    let error = BatchItemFailure::reject(BatchCode::Admission);
    assert!(run.finish::<()>(Err(error), true, true).unwrap_err().stop);
    assert_eq!(run.remaining, remaining); assert!(run.enter().is_err());
    for axis in 0..5 {
        let mut short = debit;
        match axis { 0 => short.forward_positions -= 1, 1 => short.projected_logits -= 1,
            2 => short.attention_pairs -= 1, 3 => short.projections.dot_products -= 1,
            _ => short.projections.multiply_accumulates -= 1 }
        let mut run = DocumentRunState::new(short); run.enter().unwrap();
        assert_eq!(run.charge(debit, context(1, 1)).unwrap_err().fault.code, BatchCode::WorkLimit);
        assert_eq!(run.remaining, short); assert_eq!(run.last_sequence, 0);
    }
}

#[test]
fn repeated_or_invalid_transport_sequence_cannot_reexecute_prepared_native_work() {
    for invalid in [context(0, 1), context(1, 0), BatchRequestContext {
        request_seq: 1, epoch: 1, input_line: 0, byte_offset: 0 }] {
        let mut run = DocumentRunState::new(work()); run.enter().unwrap();
        assert_eq!(run.charge(work(), invalid).unwrap_err().fault.code, BatchCode::InvalidExecution);
        assert_eq!(run.remaining, work());
    }
    let mut run = DocumentRunState::new(work()); run.enter().unwrap();
    let zero = Int8Work::default();
    run.charge(zero, context(1, 1)).unwrap(); run.finish(Ok(()), true, true).unwrap();
    run.enter().unwrap();
    assert_eq!(run.charge(zero, context(1, 2)).unwrap_err().fault.code, BatchCode::InvalidExecution);
    assert_eq!(run.remaining, work());
}

#[test]
fn unbounded_identity_ownership_refuses_before_any_clone_or_chunk_allocation() {
    let planner = planner(false); let base = identity(&planner);
    for huge_capacity in [false, true] {
        let mut changed = base.clone();
        changed.grammar_compiler_version = if huge_capacity {
            let mut value = String::with_capacity(8192); value.push('x'); value
        } else { "x".repeat(8192) };
        assert!(changed.validate().is_ok(), "identity shape alone does not bound ownership");
        let failure = Int8ClassificationDocumentBatchPlanner::validate_configuration(
            &planner, &changed, task(), planning(), mapping(), Some(&args()), work(), None).unwrap_err();
        assert_eq!(failure.code, BatchCode::InvalidLimits);
    }
    Int8ClassificationDocumentBatchPlanner::validate_configuration(
        &planner, &base, task(), planning(), mapping(), Some(&args()), work(), None).unwrap();
}
