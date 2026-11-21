//! Genuine pinned source planning and authenticated storage fixtures. No
//! native forward, semantic quality or filesystem crash proof is claimed.
use super::*;
use crate::{
    batch::source::semantic::Int4SemanticSourceAdmission,
    jobs::{FrozenManifest, InputRetentionLimits, JobContract, JobError, JobId, JobInput,
        JobSecret, MismatchField, TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork,
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, weights::projection_fixture}},
    tasks::{answer::{AnswerOptions, AnswerPassage}, ir::TaskBudget, judge::FaithfulnessPolicy,
        source_planning::grouped_int4::semantic::SourceSemanticLimits, summarize::SummaryOptions},
    tokenizer::{pinned_controls, untrusted::SafeBpeLimits},
};
use std::{cell::{Cell, RefCell}, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::{OnceLock, atomic::{AtomicU64, Ordering}}};

fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn model() -> &'static Int4ModelIdentity {
    static MODEL: OnceLock<Int4ModelIdentity> = OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(32).0.identity().clone())
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 128, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn compact() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 65_536, max_output_tokens: 8192,
        max_pair_checks: 4_000_000, max_pair_bytes: 16_000_000 }
}
fn config(answer: bool) -> Int4SemanticSourceJobConfig {
    Int4SemanticSourceJobConfig {
        task: if answer { Int4SemanticSourceBatchTask::Answer { options: AnswerOptions {
            max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: budget() } }
        else { Int4SemanticSourceBatchTask::Summarize { options: SummaryOptions { max_bullets: 2,
            max_bullet_scalars: 64, max_citations_per_bullet: 2, max_quote_scalars: 32 }, budget: budget() } },
        planning: SourcePlanningLimits { max_input_bytes: 8192, ..SourcePlanningLimits::default() },
        options: Int4SourceSemanticOptions { schema_version: 1, experimental_opt_in: true,
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 500_000, minimum_margin_milli: 100,
                evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 },
            judge_budget: budget(), limits: SourceSemanticLimits { max_output_bytes: 1 << 20,
                ..SourceSemanticLimits::default() }, max_model_work: work(1 << 56) },
        judge_planning: JudgeLimits::default(), encoding: UntrustedEncoding::ByteFallback,
        work: Int4SemanticSourceBatchLimits { max_model_work: work(1 << 58),
            masks: ExtractionMaskBudget { per_mask: MaskWorkLimits::default(),
                max_visits_per_item: 1_000_000_000, max_visits_per_run: 4_000_000_000 } },
    }
}
fn planner(answer: bool) -> Int4SemanticSourceJobPlanner {
    Int4SemanticSourceJobPlanner::for_model(model(), config(answer)).unwrap()
}
fn limits(c: &Int4SemanticSourceJobConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: c.options.limits.max_output_bytes as usize,
        max_spool_bytes: 8 << 20, max_materialized_bytes: 8 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(c.work.max_model_work, c.work.masks.max_visits_per_run) }
}
fn quote(c: &Int4SemanticSourceJobConfig) -> JobWork {
    JobWork::from_int4(c.options.max_model_work, c.work.masks.max_visits_per_item)
}
fn inputs(answer: bool) -> [JobInput<'static>; 2] {
    let (a, b): (&[u8], &[u8]) = if answer {
        (br#"{"id":"alpha","text":"Who met Bob?","task_args":{"passages":[{"id":"p1","text":"Alice met Bob."},{"id":"p2","text":"A second original passage."}]}}"#,
         br#"{"id":"beta","text":"Who met Alice?","task_args":{"passages":[{"id":"p2","text":"Bob met Alice."}]}}"#)
    } else {
        (br#"{"id":"alpha","text":"Alice met Bob."}"#,
         br#"{"id":"beta","text":"Bob met Alice.","task_args":null}"#)
    };
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn document(answer: bool) -> BatchDocument<Int4SemanticSourceBatchArgs> {
    BatchDocument { id: "alpha".to_owned(), text: if answer { "Who met Bob?" } else {
        "Alice met Bob. Café 上海 <tool_call>." }.to_owned(),
        task_args: answer.then(|| Int4SemanticSourceBatchArgs { passages: vec![
            AnswerPassage { id: "p1".to_owned(), text: "Alice met Bob. Café 上海 <tool_call>.".to_owned() }] }) }
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn freeze(p: &Int4SemanticSourceJobPlanner) -> FrozenManifest {
    let answer = matches!(p.config().task, Int4SemanticSourceBatchTask::Answer { .. });
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: p.execution_identity(),
        recipe: &p.recipe, limits: limits(&config(answer)) }, inputs(answer), &mut Continue).unwrap()
}
fn axis(work: &mut JobWork, n: usize) -> &mut u64 {
    match n { 0 => &mut work.model.forward_positions, 1 => &mut work.model.projected_logits,
        2 => &mut work.model.attention_pairs, 3 => &mut work.model.projections.dot_products,
        4 => &mut work.model.projections.multiply_accumulates, _ => &mut work.mask_node_visits }
}

#[test]
fn recipes_bind_actual_compilers_complete_qa_input_and_explicit_review_policy() {
    for answer in [false, true] {
        let p = planner(answer); let recipe = serde_json::to_value(&p.recipe).unwrap();
        assert_eq!(recipe["settings"]["task"], serde_json::to_value(config(answer).task).unwrap());
        assert_eq!(recipe["settings"]["options"], serde_json::to_value(config(answer).options).unwrap());
        assert_eq!(recipe["source_identity"], serde_json::to_value(p.execution_identity()).unwrap());
        assert_eq!(recipe["judge_identity"], serde_json::to_value(p.judge_identity()).unwrap());
        assert_eq!(recipe["encoding_scope"], ENCODING_SCOPE);
        assert_eq!(recipe["answer_passage_layout"], ANSWER_PASSAGE_LAYOUT_VERSION);
        assert_eq!(recipe["semantic_calibration"], SOURCE_SEMANTIC_CALIBRATION);
        assert_eq!(recipe["input_policy"], INPUT_POLICY); assert_eq!(recipe["group_width"], 32);
        if answer {
            assert_eq!(recipe["settings"]["answer_input"], serde_json::json!({
                "max_passages": 32, "max_input_bytes": 8192, "max_input_tokens": 8192 }));
        } else { assert!(recipe["settings"]["answer_input"].is_null()); }
        assert_eq!(p.encoding_workspace_bytes().unwrap(), 0);
        let mut c = config(answer); c.encoding = UntrustedEncoding::SafeBpeV1(compact());
        let encoded = Int4SemanticSourceJobPlanner::for_model(model(), c).unwrap();
        assert_eq!(encoded.encoding_workspace_bytes().unwrap(), compact().requirements().unwrap().workspace_bytes);
        assert_ne!(p.execution_identity().template_digest, encoded.execution_identity().template_digest);
        assert_ne!(p.judge_identity().template_digest, encoded.judge_identity().template_digest);
        let (other, _) = projection_fixture(64);
        let changed = Int4SemanticSourceJobPlanner::for_model(other.identity(), config(answer)).unwrap();
        assert_eq!(freeze(&p).binding.compare(&freeze(&changed).binding), Err(JobError::Mismatch(MismatchField::Execution)));
    }
}

#[test]
fn private_recipe_authenticates_every_source_compiler_review_and_work_setting() {
    let mut p = planner(false); let base = freeze(&p); let original = p.recipe.config.clone();
    for n in 0..58 {
        p.recipe.config = original.clone(); let c = &mut p.recipe.config;
        match n {
            0 => c.planning.max_input_bytes -= 1,
            1 => c.planning.max_context_tokens -= 1,
            2 => c.planning.max_passages -= 1,
            3 => c.planning.compiler.max_schema_bytes -= 1,
            4 => c.planning.compiler.max_string_bytes -= 1,
            5 => c.planning.compiler.max_array_items -= 1,
            6 => c.planning.compiler.max_output_bytes -= 1,
            7 => c.planning.compiler.max_states -= 1,
            8 => c.planning.compiler.max_transitions -= 1,
            9 => c.planning.compiler.max_mask_bytes -= 1,
            10 => c.planning.source.index.max_source_bytes -= 1,
            11 => c.planning.source.index.max_index_bytes -= 1,
            12 => c.planning.source.index.max_build_steps -= 1,
            13 => c.planning.source.index.max_matches -= 1,
            14 => c.planning.source.verification.max_fields -= 1,
            15 => c.planning.source.verification.max_matches -= 1,
            16 => c.planning.source.verification.max_scan_steps -= 1,
            17 => c.options.judge_policy.minimum_candidate_weight_ppm += 1,
            18 => c.options.judge_policy.minimum_margin_milli += 1,
            19 => c.options.judge_policy.evidence_window_bytes -= 1,
            20 => c.options.judge_policy.max_evidence_windows -= 1,
            21 => c.options.judge_policy.max_evidence_spans -= 1,
            22 => c.options.judge_budget.max_output_bytes -= 1,
            23 => c.options.limits.max_statements -= 1,
            24 => c.options.limits.max_source_bytes -= 1,
            25 => c.options.limits.max_claim_bytes -= 1,
            26 => c.options.limits.max_total_claim_bytes -= 1,
            27 => c.options.limits.max_total_prompt_tokens -= 1,
            28 => c.options.limits.max_total_projected_logits -= 1,
            29 => c.options.limits.max_output_bytes -= 1,
            30 => c.judge_planning.per_head.max_candidates -= 1,
            31 => c.judge_planning.per_head.max_total_tokens -= 1,
            32 => c.judge_planning.per_head.max_nodes -= 1,
            33 => c.judge_planning.per_head.max_depth -= 1,
            34 => c.judge_planning.per_head.max_candidate_id_bytes -= 1,
            35 => c.judge_planning.per_head.max_projected_logits -= 1,
            36 => c.judge_planning.max_total_prompt_tokens -= 1,
            37 => c.judge_planning.max_total_candidate_tokens -= 1,
            38 => c.judge_planning.max_total_projected_logits -= 1,
            39 => c.judge_planning.max_output_bytes -= 1,
            40 => c.work.masks.per_mask.max_trie_node_visits -= 1,
            41 => c.work.masks.per_mask.checkpoint_interval_nodes -= 1,
            42 => c.work.masks.max_visits_per_item -= 1,
            43 => c.work.masks.max_visits_per_run -= 1,
            44..=48 => { let mut value = quote(c); *axis(&mut value, n - 44) -= 1;
                c.options.max_model_work = value.int4_model_work(); },
            49..=53 => { let mut value = limits(c).max_work; *axis(&mut value, n - 49) -= 1;
                c.work.max_model_work = value.int4_model_work(); },
            54 => if let Int4SemanticSourceBatchTask::Summarize { ref mut options, .. } = c.task {
                options.max_bullet_scalars -= 1; },
            55 => if let Int4SemanticSourceBatchTask::Summarize { ref mut budget, .. } = c.task {
                budget.max_input_tokens -= 1; },
            56 => c.encoding = UntrustedEncoding::SafeBpeV1(compact()),
            _ => c.options.experimental_opt_in = false,
        }
        // Recipe-only mutation proves commitment coverage, never execution
        // authority for a mismatched or invalid hypothetical configuration.
        assert_eq!(base.binding.compare(&freeze(&p).binding), Err(JobError::Mismatch(MismatchField::Recipe)), "setting {n}");
    }
}

#[test]
fn fixed_contract_and_each_six_axis_lifetime_mismatch_refuse() {
    for answer in [false, true] {
        let p = planner(answer); let exact = limits(p.config()); p.check_job_limits(exact).unwrap();
        for n in 0..6 { for increase in [false, true] {
            let mut changed = exact; let value = axis(&mut changed.max_work, n);
            if increase { *value += 1; } else { *value -= 1; }
            assert_eq!(p.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()));
        } }
        for n in 0..3 {
            let mut changed = exact;
            match n { 0 => changed.max_result_bytes = budget().max_output_bytes as usize,
                1 => changed.max_input_bytes_per_item -= 1, _ => changed.max_id_bytes = 257 }
            assert_eq!(p.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()));
        }
        for n in 0..6 {
            let mut c = config(answer);
            match n { 0 => c.options.experimental_opt_in = false,
                1 => c.options.limits.max_output_bytes = budget().max_output_bytes - 1,
                2 => c.planning.compiler.max_schema_bytes = 1,
                3 => c.judge_planning.max_total_prompt_tokens = 0,
                4 => c.work.max_model_work.forward_positions = c.options.max_model_work.forward_positions - 1,
                _ => c.planning.max_passages = 0 }
            assert!(Int4SemanticSourceJobPlanner::for_model(model(), c).is_err(), "contract {n}");
        }
    }
}

struct NeverAdmit;
impl Int4SemanticSourceBatchAdmission for NeverAdmit {
    type Guard = ();
    fn admit(&mut self, _: Int4SemanticSourceAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        panic!("planning-only fixture reached native admission")
    }
}
fn item_error<T>(result: Result<T, BatchItemFailure>) -> BatchItemFailure {
    match result { Err(error) => error, Ok(_) => panic!("expected item refusal") }
}

#[test]
fn genuine_processor_retains_assets_quotes_full_attempt_and_refuses_foreign_seals() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Int4SemanticSourceBatchPlanner<'static>>(); send_sync::<Int4SemanticSourceJobPlanner>();
    let (weights, _) = projection_fixture(32);
    let controls = pinned_controls::pinned().unwrap();
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls()).unwrap();
    let memory = Int4MemoryRequirement::for_context(1024, 32).unwrap();
    let mut engine = StrictInt4Engine::new(&weights, 1024, Int4MemoryBudget { max_kv_bytes: memory.kv_bytes,
        max_rope_bytes: memory.rope_bytes, max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
    for answer in [false, true] { for encoding in [UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1(compact())] {
        let make = || { let mut c = config(answer); c.encoding = encoding.clone();
            Int4SemanticSourceJobPlanner::for_model(weights.identity(), c).unwrap() };
        let planner = make(); let recipe = bounded_json(&planner.recipe, SEMANTIC_SOURCE_JOB_RECIPE_BYTES).unwrap();
        let retained = planner.retain_compiler(); let weak = Arc::downgrade(&retained);
        assert!(Arc::ptr_eq(&retained, &planner.compiler)); assert_eq!(Arc::strong_count(&retained), 2);
        let mut processor = Int4SemanticSourceJobProcessor::new(planner, &mut engine, &vocabulary, NeverAdmit).unwrap();
        assert_eq!(Arc::strong_count(&retained), 2);
        assert_eq!(bounded_json(processor.job_recipe(), SEMANTIC_SOURCE_JOB_RECIPE_BYTES).unwrap(), recipe);
        let prepared = processor.prepare_with_control(document(answer), &mut Continue).unwrap();
        assert_eq!(processor.durable_work(&prepared).unwrap(), quote(&config(answer)));
        assert_eq!(processor.max_result_bytes(&prepared), config(answer).options.limits.max_output_bytes);
        assert_eq!(processor.native.reserved_model_work(), Int4Work::default());
        assert_eq!(processor.native.reserved_mask_visits(), 0);
        drop(processor);
        assert_eq!(Arc::strong_count(&retained), 1); assert!(weak.upgrade().is_some());
        assert_eq!(retained.task(), &config(answer).task);
        let other = Int4SemanticSourceJobProcessor::new(make(), &mut engine, &vocabulary, NeverAdmit).unwrap();
        assert_eq!(other.durable_work(&prepared).unwrap_err().fault.code, BatchCode::Admission);
        assert_eq!(other.native.reserved_model_work(), Int4Work::default()); drop(other);
        drop(retained); assert!(weak.upgrade().is_none());
        assert!(!engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0));
    } }
}

#[test]
fn summary_and_qa_payload_shape_and_whole_qa_input_are_checked_in_prepare() {
    let (weights, _) = projection_fixture(32); let controls = pinned_controls::pinned().unwrap();
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls()).unwrap();
    let memory = Int4MemoryRequirement::for_context(1024, 32).unwrap();
    let mut engine = StrictInt4Engine::new(&weights, 1024, Int4MemoryBudget { max_kv_bytes: memory.kv_bytes,
        max_rope_bytes: memory.rope_bytes, max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
    for answer in [false, true] {
        let mut processor = Int4SemanticSourceJobProcessor::new(planner(answer), &mut engine, &vocabulary, NeverAdmit).unwrap();
        let error = item_error(processor.prepare_with_control(document(!answer), &mut Continue));
        assert_eq!(error.fault.code, BatchCode::InvalidEnvelope); assert!(!error.stop);
        assert_eq!(processor.native.reserved_model_work(), Int4Work::default());
        let valid = processor.prepare_with_control(document(answer), &mut Continue).unwrap();
        assert_eq!(processor.durable_work(&valid).unwrap(), quote(&config(answer)));
    }
    let mut c = config(true); c.planning.max_input_bytes = 32;
    let planner = Int4SemanticSourceJobPlanner::for_model(weights.identity(), c).unwrap();
    let mut processor = Int4SemanticSourceJobProcessor::new(planner, &mut engine, &vocabulary, NeverAdmit).unwrap();
    let input = BatchDocument { id: "tiny".to_owned(), text: "Who?".to_owned(),
        task_args: Some(Int4SemanticSourceBatchArgs { passages: vec![
            AnswerPassage { id: "p".to_owned(), text: "Alice".to_owned() }] }) };
    // Ten visible source/question/id bytes still exceed 32 once the complete
    // canonical passage manifest is admitted by the unchanged source planner.
    let error = item_error(processor.prepare_with_control(input, &mut Continue));
    assert_eq!(error.fault.code, BatchCode::Planning); assert!(!error.stop);
    assert_eq!(processor.native.reserved_model_work(), Int4Work::default());
}

mod storage;
