//! Real pinned planning and private accounting fixtures. No native answers,
//! model parity, task quality or throughput evidence is manufactured here.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tasks::mapreduce::{ChunkLimits, ExecutionLimits},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls, untrusted::SafeBpeLimits},
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) } }
fn source_planner() -> Arc<SourceTaskPlanner> {
    let controls = pinned_controls::pinned().unwrap();
    Arc::new(SourceTaskPlanner::pinned(controls.template_controls(),
        EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap()).unwrap())
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private-int8-question-population-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: ANSWER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn model(id: &ExecutionIdentity) -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: id.source_revision.clone(),
        recipe_id: id.quant_recipe.clone(), source_root_sha256: Sha256Digest::of_bytes(b"fixture-source").to_hex(),
        logical_model_sha256: id.logical_model_digest.to_hex() }
}
fn config() -> Int8QuestionDocumentBatchConfig {
    let one = constrained_int8::planned_work(8128, 64).unwrap();
    let item = (0..64).try_fold(Int8Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    let verification = GroundingBudget { max_fields: 4096, max_matches: 16384, max_scan_steps: 67_108_864 };
    Int8QuestionDocumentBatchConfig { options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(8192).unwrap().kv_bytes },
        planning: SourcePlanningLimits::default(),
        mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6,
                max_chunk_tokens: 6, context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2, ..ExecutionLimits::default() },
            max_model_work: item, mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000 },
        verification, max_question_bytes: 1024, max_model_work: item.checked_add(item).unwrap(), max_mask_visits: 128_000,
        max_verification: add_verification(verification, verification).unwrap() }
}
fn factory() -> Int8QuestionDocumentBatchPlanner {
    let p = source_planner(); let id = identity(&p);
    Int8QuestionDocumentBatchPlanner::for_model(&model(&id), p, id, config()).unwrap()
}
fn prepared(f: &Int8QuestionDocumentBatchPlanner, source: &str, question: &str) -> PreparedInt8QuestionDocument {
    f.prepare_with_control(source.to_owned(), QuestionDocumentArgs { question: question.to_owned() }, &mut Continue).unwrap()
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: sequence * 128 }
}
fn native_axis(work: &mut Int8Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits, 2 => &mut work.attention_pairs,
        3 => &mut work.projections.dot_products, _ => &mut work.projections.multiply_accumulates }
}
fn rebuild(f: &Int8QuestionDocumentBatchPlanner, item: &PreparedInt8QuestionDocument) -> Result<(), BatchItemFailure> {
    f.owns(item)?;
    let plan = f.plan(&item.source, &item.question, &item.operation, &mut Continue, true)?;
    f.compare(&plan, item, &mut Continue)
}

#[test]
fn record_arguments_require_exactly_one_question_object() {
    for raw in ["null", "[]", r#"["Who?"]"#, "{}", r#"{"question":null}"#,
        r#"{"question":"Who?","question":"Where?"}"#, r#"{"question":"Who?","passages":[]}"#,
        r#"{"question":"Who?","budget":{}}"#, r#"{"question":"Who?","options":{}}"#,
        r#"{"question":"Who?","synthesize_answer":true}"#] {
        assert!(serde_json::from_str::<QuestionDocumentArgs>(raw).is_err(), "{raw}");
    }
    let record = serde_json::from_str::<BatchDocument<QuestionDocumentArgs>>(
        r#"{"id":"opaque","text":"complete original source","task_args":{"question":"Who?\n上海"}}"#).unwrap();
    assert_eq!(record.text, "complete original source"); assert_eq!(record.task_args.unwrap().question, "Who?\n上海");
}

#[test]
fn complete_source_and_per_item_question_remain_exact_reconstruction_authority() {
    let f = factory(); let item = prepared(&f, "Alice Bob Carol Dave", "Who is named?");
    rebuild(&f, &item).unwrap();
    assert!(item.chunk_count() > 1); assert_eq!(item.execution_identities().len(), item.native_chunks());
    assert_eq!(item.expected.source_span().byte_start, 0);
    assert_eq!(item.expected.source_span().byte_end, item.source.len());
    assert_eq!(item.reserved_mask_visits(), item.native_chunks() as u64 * f.config.mapping.mask_visits_per_chunk);
    for (source, question) in [(item.source.as_str(), "Where is named?"), ("Carol Bob Alice Dave", item.question.as_str())] {
        let changed = f.plan(source, question, &item.operation, &mut Continue, false).unwrap();
        assert_eq!(f.compare(&changed, &item, &mut Continue).unwrap_err().fault.code, BatchCode::Admission);
    }
    let another = factory(); assert_eq!(another.owns(&item).unwrap_err().fault.code, BatchCode::Admission);
    let mut identity_tamper = prepared(&f, "Alice Bob Carol Dave", "Who is named?");
    identity_tamper.identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"changed last passage");
    assert_eq!(rebuild(&f, &identity_tamper).unwrap_err().fault.code, BatchCode::Admission);
    let whitespace = prepared(&f, "Alice       Bob        Carol", "Who is named?");
    assert_eq!(whitespace.expected.source_span().byte_end, whitespace.source.len());
    assert!(whitespace.chunk_count() >= whitespace.native_chunks());
    assert_eq!(whitespace.reserved_mask_visits(), whitespace.native_chunks() as u64 * 1000);
}

#[test]
fn actual_model_policy_and_complete_limits_refuse_before_preparation() {
    let p = source_planner(); let id = identity(&p);
    for axis in 0..4 {
        let mut actual = model(&id);
        match axis { 0 => actual.model_id.push('x'), 1 => actual.revision.push('x'),
            2 => actual.recipe_id.push('x'), _ => actual.logical_model_sha256 = "00".repeat(32) }
        assert!(Int8QuestionDocumentBatchPlanner::for_model(&actual, Arc::clone(&p), id.clone(), config()).is_err());
    }
    for axis in 0..8 {
        let mut c = config();
        match axis { 0 => c.budget.max_kv_bytes -= 1, 1 => c.mapping.chunks.context_tokens -= 1,
            2 => c.mapping.reduction.reduce_fan_in = 1, 3 => c.mapping.mask_limits.checkpoint_interval_nodes = 0,
            4 => c.max_question_bytes = 0, 5 => c.max_mask_visits = c.mapping.max_mask_visits - 1,
            6 => c.max_verification.max_matches = c.verification.max_matches - 1,
            _ => c.options.max_citations = 0 }
        assert!(Int8QuestionDocumentBatchPlanner::for_model(&model(&id), Arc::clone(&p), id.clone(), c).is_err(), "axis {axis}");
    }
    let f = factory();
    for (source, question) in [("", "Who?"), ("Alice", ""), ("Alice", " \r\n")] {
        assert!(f.prepare_with_control(source.to_owned(), QuestionDocumentArgs { question: question.to_owned() }, &mut Continue).is_err());
    }
}

#[test]
fn native_mask_and_verification_debits_are_atomic_and_never_reset_at_flush() {
    let f = factory(); let item = prepared(&f, "Alice Bob", "Who?"); let expected = item.expected;
    let mut config = f.config;
    config.max_model_work = item.planned_work().checked_add(item.planned_work()).unwrap();
    config.max_mask_visits = item.reserved_mask_visits() * 2;
    for axis in 0..9 {
        let mut limited = config;
        match axis { 0..=4 => *native_axis(&mut limited.max_model_work, axis) -= 1,
            5 => limited.max_mask_visits -= 1, 6 => limited.max_verification.max_fields -= 1,
            7 => limited.max_verification.max_matches -= 1, _ => limited.max_verification.max_scan_steps -= 1 }
        let mut ledger = Ledger::default(); ledger.begin(&limited, expected, context(1, 1)).unwrap();
        let prior = (ledger.work, ledger.masks, ledger.verification); ledger.failed = false;
        assert_eq!(ledger.begin(&limited, expected, context(2, 2)).unwrap_err().fault.code, BatchCode::WorkLimit);
        assert_eq!((ledger.work, ledger.masks, ledger.verification), prior);
        assert!(ledger.failed); assert!(ledger.ready().is_err());
    }
    let mut ledger = Ledger::default(); ledger.begin(&config, expected, context(1, 1)).unwrap();
    ledger.failed = false; ledger.begin(&config, expected, context(2, 2)).unwrap();
    assert_eq!(ledger.work, config.max_model_work); assert_eq!(ledger.masks, config.max_mask_visits);
    assert_eq!(ledger.verification, config.max_verification);
    ledger.failed = false; assert!(ledger.begin(&config, expected, context(3, 3)).is_err());
}

#[test]
fn stale_sequences_and_all_original_cancellation_causes_remain_fatal() {
    let f = factory(); let item = prepared(&f, "Alice Bob", "Who?");
    for next in [context(1, 2), context(2, 0), BatchRequestContext { input_line: 1, ..context(2, 2) }] {
        let mut ledger = Ledger::default(); ledger.begin(&f.config, item.expected, context(1, 1)).unwrap();
        ledger.failed = false;
        assert_eq!(ledger.begin(&f.config, item.expected, next).unwrap_err().fault.code, BatchCode::InvalidExecution);
        assert_eq!(ledger.work, item.planned_work());
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = f.prepare_with_control("PRIVATE SOURCE Alice".to_owned(),
            QuestionDocumentArgs { question: "PRIVATE QUESTION Who?".to_owned() }, &mut Stop(cause)).err().unwrap();
        assert_eq!(error.fault, BatchFault::cancelled(cause)); assert!(error.stop);
        let mapped = map_failure(Int8SourceError::Cancelled(cause).into(), false);
        assert_eq!(mapped.fault, BatchFault::cancelled(cause)); assert!(mapped.stop);
        let display = format!("{}", error.fault);
        assert!(!display.contains("PRIVATE SOURCE")); assert!(!display.contains("PRIVATE QUESTION"));
    }
}

fn compact_policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn allowance() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }
}
fn used(item: &PreparedInt8QuestionDocument) -> SafeBpeWork { item.map_document_encoding_work().unwrap().unwrap() }

#[test]
fn source_question_probes_and_exact_rebuild_share_one_compact_operation() {
    let f = factory().with_map_document_encoding(compact_policy(), allowance()).unwrap();
    let source = "Alice met Bob in 上海. Carol saw Alice. <think>"; let question = "Who met Alice? <|im_start|>";
    let first = prepared(&f, source, question); let second = prepared(&f, source, question);
    let once = used(&first);
    assert!(once.input_bytes() > source.len() + question.len());
    assert!(!Arc::ptr_eq(&first.operation, &second.operation)); assert_eq!(used(&second), once);
    rebuild(&f, &first).unwrap(); assert_eq!(used(&first), once.checked_add(once).unwrap());
    assert_eq!(used(&second), once); assert_eq!(f.planner.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
    let old = prepared(&f, source, question);
    let changed = f.with_map_document_encoding(compact_policy(), allowance()).unwrap();
    assert!(changed.owns(&old).is_err());
}

#[test]
fn all_four_compact_work_caps_cover_both_compilations_and_refusal_retains_consumption() {
    let source = "Alice met Bob in 上海. Carol saw Alice."; let question = "Who met Alice?";
    let probe = factory().with_map_document_encoding(compact_policy(), allowance()).unwrap();
    let item = prepared(&probe, source, question); let once = used(&item); let twice = once.checked_add(once).unwrap();
    let total = SafeBpeWorkLimits { max_input_bytes: twice.input_bytes() as u64,
        max_output_tokens: twice.output_tokens() as u64, max_pair_checks: twice.pair_checks(), max_pair_bytes: twice.pair_bytes() };
    let exact = factory().with_map_document_encoding(compact_policy(), total).unwrap();
    let item = prepared(&exact, source, question); rebuild(&exact, &item).unwrap();
    assert_eq!(used(&item), twice); assert!(rebuild(&exact, &item).is_err()); assert_eq!(used(&item), twice);
    for axis in 0..4 {
        let mut short = total;
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        let limited = factory().with_map_document_encoding(compact_policy(), short).unwrap();
        let item = prepared(&limited, source, question); assert_eq!(used(&item), once);
        assert!(rebuild(&limited, &item).is_err()); let consumed = used(&item);
        assert!(rebuild(&limited, &item).is_err()); assert!(used(&item).input_bytes() >= consumed.input_bytes());
        assert_eq!(used(&prepared(&limited, source, question)), once);
    }
}
