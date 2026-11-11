//! Real pinned planning and private accounting fixtures. No native output,
//! model quality, inference parity or throughput is fabricated by these tests.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION, canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{Int4MemoryBudget, STRICT_INT4_EXECUTION}},
    tasks::mapreduce::{ChunkLimits, ExecutionLimits},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};
const WIDTH: usize = 127;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) } }
fn planner() -> Arc<SourceTaskPlanner> {
    let controls = pinned_controls::pinned().unwrap();
    Arc::new(SourceTaskPlanner::pinned(controls.template_controls(),
        EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap()).unwrap())
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private-question-corpus-metadata-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH)).unwrap()),
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: ANSWER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn config(synthesis: bool) -> Int4QuestionDocumentBatchConfig {
    let one = constrained_int4::planned_work(WIDTH, 8128, 64).unwrap();
    let mut item = one;
    for _ in 0..8 { item = item.checked_add(item).unwrap(); }
    let verification = GroundingBudget { max_fields: 4096, max_matches: 16384, max_scan_steps: 67_108_864 };
    Int4QuestionDocumentBatchConfig { options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes },
        planning: SourcePlanningLimits::default(),
        mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2, ..ExecutionLimits::default() },
            max_model_work: item, mask_limits: MaskWorkLimits::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 256000, verification },
        verification, synthesis: synthesis.then_some(QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 4096 }),
        hierarchy: None,
        max_question_bytes: 1024, max_model_work: item.checked_add(item).unwrap(), max_mask_visits: 512000,
        max_map_verification: add_verification(verification, verification).unwrap(),
        max_verification: add_verification(verification, verification).unwrap() }
}
fn factory(synthesis: bool) -> Int4QuestionDocumentBatchPlanner {
    let p = planner(); Int4QuestionDocumentBatchPlanner::configured(Arc::clone(&p), identity(&p), WIDTH, config(synthesis)).unwrap()
}
fn prepared(factory: &Int4QuestionDocumentBatchPlanner, source: &str, question: &str) -> PreparedInt4QuestionDocument {
    factory.prepare_with_control(source.to_owned(),
        QuestionDocumentArgs { question: question.to_owned() }, &mut Continue).unwrap()
}
fn context(seq: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: seq, byte_offset: seq * 128 }
}
fn native_axis(work: &mut Int4Work, n: usize) -> &mut u64 {
    match n { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits, 2 => &mut work.attention_pairs,
        3 => &mut work.projections.dot_products, _ => &mut work.projections.multiply_accumulates }
}
fn budget(factory: &Int4QuestionDocumentBatchPlanner) -> Int4SourceMapCohortBudget {
    let need = factory.cohort_requirements().unwrap();
    Int4SourceMapCohortBudget { context_limit: factory.config.planning.max_context_tokens,
        native_memory: Int4MemoryBudget { max_kv_bytes: need.native.kv_bytes, max_rope_bytes: need.native.rope_bytes,
            max_scratch_payload_bytes: need.native.scratch_payload_bound },
        max_workspace_bytes: need.workspace_bytes, max_result_bytes: need.max_result_bytes }
}
#[test]
fn record_arguments_are_only_a_required_exact_question_object() {
    for raw in ["null", "[]", r#"["who?"]"#, "{}", r#"{"question":null}"#,
        r#"{"question":"who?","question":"where?"}"#, r#"{"question":"who?","passages":[]}"#,
        r#"{"question":"who?","budget":{}}"#, r#"{"question":"who?","options":{}}"#,
        r#"{"question":"who?","synthesize_answer":true}"#] {
        assert!(serde_json::from_str::<QuestionDocumentArgs>(raw).is_err(), "{raw}");
    }
    let args = serde_json::from_str::<QuestionDocumentArgs>(r#"{"question":"Who?\n上海"}"#).unwrap();
    assert_eq!(args.question, "Who?\n上海");
    let record = serde_json::from_str::<BatchDocument<QuestionDocumentArgs>>(
        r#"{"id":"opaque","text":"the entire original source","task_args":{"question":"Who?"}}"#).unwrap();
    assert_eq!(record.text, "the entire original source"); assert_eq!(record.task_args.unwrap().question, "Who?");
}
#[test]
fn private_question_and_source_changes_invalidate_every_real_native_plan() {
    for synthesis in [false, true] {
        let f = factory(synthesis); let original = prepared(&f, "Alice Bob Carol Dave", "Who is named?");
        let same = f.plan(&original.source, &original.question, &mut Continue, false).unwrap();
        f.compare(&same, &original, &mut Continue).unwrap();
        for (source, question) in [(original.source.as_str(), "Where is named?"), ("Carol Bob Alice Dave", original.question.as_str())] {
            let changed = f.plan(source, question, &mut Continue, false).unwrap();
            let refused = f.compare(&changed, &original, &mut Continue).unwrap_err();
            assert!(refused.stop); assert_eq!(refused.fault.code, BatchCode::Admission);
        }
        assert!(original.execution_identities().iter().all(|id| id.task_spec == ANSWER_TASK_VERSION));
        assert_eq!(original.planned_work().projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
    }
}
#[test]
fn logical_whitespace_ranges_have_no_fabricated_native_rows_or_mask_debits() {
    for synthesis in [false, true] {
        let f = factory(synthesis); let p = prepared(&f, "ééé                        ééé", "Who is named?");
        assert!(p.chunk_count() > p.native_chunks());
        assert_eq!(p.execution_identities().len(), p.native_chunks());
        assert_eq!(p.reserved_mask_visits(), (p.native_chunks() as u64 + u64::from(synthesis)) * 1000);
        assert_eq!(p.expected.discovery().source_span().byte_end, p.source.len());
        assert_eq!(p.expected.discovery().source_span().scalar_end, p.source.chars().count());
        assert!(f.plan("                        ", "Who is named?", &mut Continue, false).is_err());
    }
}
#[test]
fn native_schedule_changes_rotate_the_private_preparation_owner() {
    let f = factory(false); let p = prepared(&f, "Alice Bob", "Who is named?");
    let unrelated = factory(false);
    assert_eq!(unrelated.owns(&p).unwrap_err().fault.code, BatchCode::Admission);
    let grouped = f.with_chunk_cohorts().unwrap();
    assert!(grouped.owns(&p).is_err());
    let current = prepared(&grouped, "Alice Bob", "Who is named?");
    grouped.owns(&current).unwrap();
    grouped.with_chunk_cohorts().unwrap().owns(&current).unwrap();
}
#[test]
fn flat_synthesis_reserves_the_full_scalar_pass_before_any_discovery() {
    let serial = factory(false); let synthesized = factory(true);
    let plain = prepared(&serial, "Alice Bob", "Who is named?");
    let combined = prepared(&synthesized, "Alice Bob", "Who is named?");
    let reserve = constrained_int4::planned_work(WIDTH, 8128, 64).unwrap();
    assert_eq!(combined.planned_work(), plain.planned_work().checked_add(reserve).unwrap());
    assert_eq!(combined.reserved_mask_visits(), plain.reserved_mask_visits() + 1000);
    assert_eq!(combined.execution_identities(), plain.execution_identities());
    let mut insufficient = config(true); insufficient.mapping.max_model_work = Int4Work::default();
    assert!(check_synthesis_reserve(&insufficient, WIDTH).is_err());
    let mut insufficient = config(true); insufficient.mapping.max_mask_visits = 1000;
    assert!(check_synthesis_reserve(&insufficient, WIDTH).is_err());
}
fn hierarchy_factory(passes: usize) -> Int4QuestionDocumentBatchPlanner {
    let p = planner(); let mut config = config(true);
    config.hierarchy = Some(QuestionHierarchyLimits { max_passes: passes, ..QuestionHierarchyLimits::default() });
    Int4QuestionDocumentBatchPlanner::configured(Arc::clone(&p), identity(&p), WIDTH, config).unwrap()
}
#[test]
fn each_private_question_hierarchy_reserves_all_passes_before_discovery() {
    let plain = prepared(&factory(false), "Alice Bob Carol", "Who is named?");
    let pass = constrained_int4::planned_work(WIDTH, 8128, 64).unwrap();
    for passes in [1, 3, 16] {
        let factory = hierarchy_factory(passes);
        let sealed = factory.prepare_with_control("Alice Bob Carol".to_owned(),
            QuestionDocumentArgs { question: "Who is named?".to_owned() }, &mut Continue).unwrap();
        let mut expected = plain.planned_work();
        for _ in 0..passes { expected = expected.checked_add(pass).unwrap(); }
        assert_eq!(sealed.planned_work(), expected);
        assert_eq!(sealed.reserved_mask_visits(), plain.reserved_mask_visits() + passes as u64 * 1000);
        assert_eq!(sealed.execution_identities(), plain.execution_identities());
        assert_eq!(factory.max_output_tokens(), passes as u64 * 64);
        let same = factory.plan(&sealed.source, &sealed.question, &mut Continue, false).unwrap();
        factory.compare(&same, &sealed, &mut Continue).unwrap();
        let changed = factory.plan(&sealed.source, "Where is named?", &mut Continue, false).unwrap();
        assert_eq!(factory.compare(&changed, &sealed, &mut Continue).unwrap_err().fault.code, BatchCode::Admission);
        let stopped = factory.plan(&sealed.source, &sealed.question, &mut Stop(DecodeCancellationKind::Deadline), false)
            .err().unwrap();
        assert_eq!(stopped.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    }
}
#[test]
fn hierarchy_and_discovery_schedules_are_independent_and_sealed_to_prepared_ownership() {
    let f = hierarchy_factory(3); let old = prepared(&f, "Alice Bob", "Who is named?");
    let grouped = f.with_hierarchy_cohorts(2).unwrap();
    assert!(grouped.uses_native_cohorts()); assert!(!grouped.uses_chunk_cohorts());
    assert!(grouped.owns(&old).is_err());
    assert_eq!(grouped.hierarchy_cohort_schedule(), Some(Int4HierarchyCohortSchedule {
        max_groups: 2, discovery_cohorts: false }));
    assert_eq!(grouped.cohort_requirements().unwrap().native.kv_bytes, grouped.config.budget.max_kv_bytes * 2);
    let current = prepared(&grouped, "Alice Bob", "Who is named?");
    let grouped = grouped.with_hierarchy_cohorts(2).unwrap(); grouped.owns(&current).unwrap();
    let both = grouped.with_chunk_cohorts().unwrap();
    assert!(both.owns(&current).is_err());
    assert_eq!(both.hierarchy_cohort_schedule().unwrap().discovery_cohorts, true);
    assert_eq!(both.cohort_requirements().unwrap().native.kv_bytes, both.config.budget.max_kv_bytes * 3);
    let exact = budget(&both); backend::cohort_budget(&both, exact).unwrap();
    let plan = both.plan("Alice", "Who is named?", &mut Continue, false).unwrap();
    let actual = plan.phase_requirements(exact.context_limit, both.hierarchy_cohort_schedule()).unwrap();
    assert!(actual.native.kv_bytes <= exact.native_memory.max_kv_bytes);
    for groups in [0, 4, 65, usize::MAX] { assert!(hierarchy_factory(3).with_hierarchy_cohorts(groups).is_err()); }
    assert!(factory(true).with_hierarchy_cohorts(1).is_err());
    assert!(factory(false).with_hierarchy_cohorts(1).is_err());
}
#[test]
fn hierarchy_refuses_missing_policy_or_insufficient_whole_pass_authority() {
    let mut missing = config(false); missing.hierarchy = Some(QuestionHierarchyLimits::default());
    assert_eq!(missing.validate().unwrap_err().code, BatchCode::InvalidLimits);
    let f = hierarchy_factory(3); let plan = prepared(&f, "Alice Bob", "Who is named?");
    for axis in 0..6 {
        let mut c = f.config;
        let mut required = constrained_int4::planned_work(WIDTH, 8128, 64).unwrap();
        required = required.checked_add(required).unwrap().checked_add(
            constrained_int4::planned_work(WIDTH, 8128, 64).unwrap()).unwrap();
        if axis < 5 { *native_axis(&mut c.mapping.max_model_work, axis) = *native_axis(&mut required, axis) - 1; }
        else { c.mapping.max_mask_visits = 3000; }
        assert_eq!(check_synthesis_reserve(&c, WIDTH).unwrap_err().code, BatchCode::InvalidLimits);
    }
    let mut ledger = Ledger::default(); ledger.begin(&f.config, plan.expected, context(1, 1)).unwrap();
    assert_eq!(ledger.verification, f.config.verification, "one shared QA pool spans every level");
    assert!(ledger.ready().is_err(), "failure cannot refund hierarchy reservations");
    ledger.failed = false; ledger.begin(&f.config, plan.expected, context(4, 2)).unwrap();
    assert_eq!(ledger.work, plan.planned_work().checked_add(plan.planned_work()).unwrap());
    ledger.failed = false;
    assert_eq!(ledger.begin(&f.config, plan.expected, context(8, 3)).unwrap_err().fault.code, BatchCode::WorkLimit);
}
#[test]
fn all_native_mask_and_both_verification_axes_reserve_atomically() {
    let f = factory(true); let p = prepared(&f, "Alice Bob", "Who is named?");
    for axis in 0..12 {
        let mut c = f.config; let mut work = p.planned_work();
        match axis {
            0..=4 => *native_axis(&mut c.max_model_work, axis) = *native_axis(&mut work, axis) - 1,
            5 => c.max_mask_visits = p.reserved_mask_visits() - 1,
            6 => c.max_map_verification.max_fields = c.mapping.verification.max_fields - 1,
            7 => c.max_map_verification.max_matches = c.mapping.verification.max_matches - 1,
            8 => c.max_map_verification.max_scan_steps = c.mapping.verification.max_scan_steps - 1,
            9 => c.max_verification.max_fields = c.verification.max_fields - 1,
            10 => c.max_verification.max_matches = c.verification.max_matches - 1,
            _ => c.max_verification.max_scan_steps = c.verification.max_scan_steps - 1,
        }
        let mut ledger = Ledger::default();
        let error = ledger.begin(&c, p.expected, context(1, 1)).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(ledger.work, Int4Work::default()); assert_eq!(ledger.masks, 0);
        assert_eq!(ledger.map_verification, Ledger::default().map_verification);
        assert_eq!(ledger.verification, Ledger::default().verification); assert_eq!(ledger.sequence, 0);
    }
}
#[test]
fn failure_and_flush_epochs_never_refund_or_renew_attempt_allowances() {
    let f = factory(true); let p = prepared(&f, "Alice Bob", "Who is named?"); let mut ledger = Ledger::default();
    ledger.begin(&f.config, p.expected, context(1, 1)).unwrap();
    assert!(ledger.ready().is_err());
    assert_eq!(ledger.work, p.planned_work()); assert_eq!(ledger.verification, f.config.verification);
    // Successful completion clears only the fail-stop latch, not any counter.
    ledger.failed = false;
    ledger.begin(&f.config, p.expected, context(4, 2)).unwrap();
    assert_eq!(ledger.work, p.planned_work().checked_add(p.planned_work()).unwrap());
    assert_eq!(ledger.masks, 2 * p.reserved_mask_visits());
    assert_eq!(ledger.verification, f.config.max_verification); assert_eq!(ledger.map_verification, f.config.max_map_verification);
    ledger.failed = false;
    assert_eq!(ledger.begin(&f.config, p.expected, context(8, 3)).unwrap_err().fault.code, BatchCode::WorkLimit);
}
#[test]
fn sequence_replay_backward_epoch_or_line_cannot_authorize_an_attempt() {
    let f = factory(false); let p = prepared(&f, "Alice Bob", "Who is named?");
    for axis in 0..4 {
        let mut ledger = Ledger::default(); ledger.begin(&f.config, p.expected, context(3, 2)).unwrap(); ledger.failed = false;
        let mut next = context(5, 3);
        match axis { 0 => next.request_seq = 3, 1 => next.epoch = 0, 2 => next.epoch = 1, _ => next.input_line = 3 }
        assert_eq!(ledger.begin(&f.config, p.expected, next).unwrap_err().fault.code, BatchCode::InvalidExecution);
        assert_eq!(ledger.work, p.planned_work());
    }
}
#[test]
fn declared_cohort_peak_cannot_be_underpriced_by_a_short_or_sparse_document() {
    for synthesis in [false, true] {
        let f = factory(synthesis).with_chunk_cohorts().unwrap(); let exact = budget(&f);
        let declared = backend::cohort_budget(&f, exact).unwrap();
        let p = f.plan("Alice", "Who is named?", &mut Continue, false).unwrap();
        let actual = p.cohort_requirements(exact.context_limit).unwrap();
        assert!(actual.native.kv_bytes <= declared.native.kv_bytes);
        assert_eq!(declared.native.kv_bytes, 3 * f.config.budget.max_kv_bytes);
        for axis in 0..7 {
            let mut short = exact;
            match axis { 0 => short.native_memory.max_kv_bytes -= 1, 1 => short.native_memory.max_rope_bytes -= 1,
                2 => short.native_memory.max_scratch_payload_bytes -= 1, 3 => short.max_workspace_bytes -= 1,
                4 => short.max_result_bytes -= 1, 5 => short.context_limit -= 1,
                _ => short.max_result_bytes = crate::batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES + 1 }
            assert_eq!(backend::cohort_budget(&f, short).unwrap_err().fault.code, BatchCode::Admission);
        }
        assert!(backend::cohort_budget(&factory(synthesis), exact).is_err());
    }
}
#[test]
fn record_byte_limits_cancellation_and_metadata_bound_precede_native_authority() {
    let f = factory(false);
    for (source, question) in [("", "Who?"), ("Alice", ""), ("Alice", " \r\n"),
        ("Alice", &"x".repeat(1025)), (&"x".repeat(4097), "Who?")] {
        let refusal = f.plan(source, question, &mut Continue, false).err().unwrap(); assert!(!refusal.stop);
    }
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota] {
        let refusal = f.plan("Alice", "Who?", &mut Stop(cause), false).err().unwrap();
        assert!(refusal.stop); assert_eq!(refusal.fault.cancellation, Some(cause));
    }
    let mut oversized = f.identity.clone(); oversized.sampler_version = "x".repeat(MAX_QUESTION_IDENTITY_BYTES);
    assert_eq!(check_identity_size(&oversized).unwrap_err().code, BatchCode::InvalidLimits);
    let mut c = config(false); c.max_verification.max_fields = 2_000_000;
    c.max_map_verification.max_matches = 2_000_000; c.validate().unwrap();
}

mod compact;
