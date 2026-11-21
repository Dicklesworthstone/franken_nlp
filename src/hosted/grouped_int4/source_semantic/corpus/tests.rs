//! Pure memory/admission checks; no inference or quality qualification.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchLimits, BatchWork, extract::ExtractionMaskBudget},
    native_engine::{constrained_int4::planned_work, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE}},
    tasks::{answer::AnswerOptions, judge::FaithfulnessPolicy, summarize::SummaryOptions},
    tokenizer::untrusted::SafeBpeLimits,
};

const WIDTH: usize = 32;
const MIB: u64 = 1 << 20;

fn work() -> Int4Work {
    Int4Work { forward_positions: u64::MAX / 8, projected_logits: u64::MAX / 8,
        attention_pairs: u64::MAX / 8, projections: ProjectionWork {
            dot_products: u64::MAX / 8, multiply_accumulates: u64::MAX / 8 } }
}
fn fixture(answer: bool) -> Int4SourceSemanticCorpusConfig {
    let kv = Int4MemoryRequirement::for_context(256, WIDTH).unwrap().kv_bytes;
    let source = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: kv };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 4096, max_kv_bytes: kv };
    let semantic = SourceSemanticLimits { max_statements: 3, max_source_bytes: 32_768,
        max_claim_bytes: 128, max_total_claim_bytes: 1024, max_total_prompt_tokens: 2048,
        max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 };
    let run_work = work().checked_add(work()).unwrap();
    let task = if answer {
        Int4SemanticSourceBatchTask::Answer { options: AnswerOptions::default(), budget: source }
    } else {
        Int4SemanticSourceBatchTask::Summarize { options: SummaryOptions {
            max_bullets: 3, max_bullet_scalars: 32, max_citations_per_bullet: 2, max_quote_scalars: 32,
        }, budget: source }
    };
    Int4SourceSemanticCorpusConfig {
        task, planning: SourcePlanningLimits { max_input_bytes: 1024,
            max_context_tokens: 256, max_passages: 4, ..SourcePlanningLimits::default() },
        options: Int4SourceSemanticOptions { schema_version: 1, experimental_opt_in: true,
            judge_budget: judge, judge_policy: FaithfulnessPolicy {
                minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0, evidence_window_bytes: 4096,
                max_evidence_windows: 8, max_evidence_spans: 8 },
            limits: semantic, max_model_work: work() },
        judge_planning: JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
            max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() },
        encoding: UntrustedEncoding::ByteFallback,
        work: Int4SemanticSourceBatchLimits { max_model_work: run_work,
            masks: ExtractionMaskBudget {
                per_mask: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
                max_visits_per_item: 32_000, max_visits_per_run: 128_000 } },
        limits: CorpusLimits {
            native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: MIB,
                run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 1_000_000,
                    cleanup_reserve_bytes: 65_536 } },
            transport: BatchLimits { max_line_bytes: 4096, max_document_bytes: 1024,
                max_output_line_bytes: 135_168, max_output_bytes: 8 * MIB,
                max_work: BatchWork { forward_positions: run_work.forward_positions,
                    projected_logits: run_work.projected_logits }, ..BatchLimits::default() },
            preparation_reserve_bytes: 512 * MIB, io_reserve_bytes: 65_536,
        },
    }
}
fn base(config: &Int4SourceSemanticCorpusConfig) -> Int4SourceSemanticHostRequirements {
    semantic_host(config.work, config.limits).requirements_with_encoding(WIDTH, config.task.budget(),
        config.options.judge_budget, config.options.limits, config.judge_planning,
        config.input_ceiling().unwrap(), &config.encoding).unwrap()
}
fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"semantic-source-corpus-admission-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, WIDTH)).unwrap()),
        tokenizer_digest: digest, template_digest: digest, task_spec: task.to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "fixture".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn request<'a>(config: &'a Int4SourceSemanticCorpusConfig, source: &'a ExecutionIdentity,
    judge: &'a ExecutionIdentity) -> Int4SemanticSourceAdmission<'a> {
    let required = config.requirements(WIDTH).unwrap();
    Int4SemanticSourceAdmission {
        identity: source, judge_identity: judge, group_width: WIDTH,
        model_work: config.options.max_model_work, source_reserved_work: planned_work(WIDTH, 64, 8).unwrap(),
        mask_node_visits: config.work.masks.max_visits_per_item,
        mask_limits: config.work.masks.per_mask, kv_reservation_bytes: required.native.kv_bytes,
        max_result_bytes: config.options.limits.max_output_bytes, source_budget: config.task.budget(),
        judge_budget: config.options.judge_budget, semantic_limits: config.options.limits,
        judge_planning_limits: config.judge_planning, input_facts: config.input_ceiling().unwrap(),
        input_encoding: &config.encoding,
    }
}

#[test]
fn both_source_tasks_add_transport_once_to_the_complete_review_peak() {
    for answer in [false, true] {
        let config = fixture(answer);
        let one = base(&config);
        let corpus = config.requirements(WIDTH).unwrap();
        assert_eq!(corpus.preparation_bytes, one.preparation_bytes);
        assert_eq!(corpus.input_bytes, one.preparation_bytes + config.limits.reservation_bytes().unwrap()
            - config.limits.preparation_reserve_bytes);
        assert_eq!(corpus.native, one.native);
        assert_eq!(corpus.workspace_bytes, one.native_workspace_bytes);
        assert_eq!(corpus.output_reservation_bytes, one.output_bytes);
        assert_eq!(corpus.team_workspace_bytes, 0);
        assert_eq!(corpus.total_bytes().unwrap(), corpus.input_bytes + corpus.native.kv_bytes
            + corpus.workspace_bytes + corpus.output_reservation_bytes);
        assert!(corpus.preparation_bytes > config.limits.preparation_reserve_bytes);
        assert!(corpus.output_reservation_bytes > 4 * config.options.limits.max_output_bytes);
        assert_eq!(corpus, int4_source_semantic_corpus_requirements(WIDTH, config.task.budget(),
            &config.options, config.judge_planning, config.input_ceiling().unwrap(),
            &config.encoding, config.work, config.limits, None).unwrap());
    }
}

#[test]
fn qa_prices_question_and_every_passage_separately_from_complete_source_text() {
    let summary = fixture(false).requirements(WIDTH).unwrap();
    let mut config = fixture(true);
    let qa = config.requirements(WIDTH).unwrap();
    assert_eq!(qa.preparation_bytes - summary.preparation_bytes, 8 * 128 + 4 * 2048);
    assert_eq!(qa.native, summary.native);
    assert_eq!(qa.output_reservation_bytes, summary.output_reservation_bytes);
    config.options.limits.max_claim_bytes *= 2;
    config.planning.max_passages += 1;
    let expanded = config.requirements(WIDTH).unwrap();
    assert_eq!(expanded.preparation_bytes - qa.preparation_bytes, 8 * 128 + 2048);
    assert_eq!(config.input_ceiling().unwrap().source_bytes, config.planning.max_input_bytes);
    config.options.limits.max_source_bytes = 512;
    assert_eq!(config.input_ceiling().unwrap().source_bytes, 512);
}

#[test]
fn whole_run_totals_do_not_multiply_one_document_geometry_or_allow_input_scope_mismatch() {
    let mut config = fixture(false);
    let before = config.requirements(WIDTH).unwrap();
    config.work.max_model_work = config.work.max_model_work.checked_add(config.work.max_model_work).unwrap();
    config.work.masks.max_visits_per_run *= 2;
    config.limits.transport.max_requests *= 2;
    config.limits.transport.max_input_bytes *= 2;
    assert_eq!(config.requirements(WIDTH).unwrap(), before);
    config.limits.transport.max_document_bytes *= 2;
    assert!(config.requirements(WIDTH).is_err());
    config.planning.max_input_bytes *= 2;
    let larger = config.requirements(WIDTH).unwrap();
    assert_eq!(larger.native, before.native);
    assert_eq!(larger.preparation_bytes - before.preparation_bytes,
        (config.options.limits.max_statements as u64 + 4) * 1024);
}

#[test]
fn compact_encoder_and_persistent_team_have_single_independent_charges() {
    for answer in [false, true] {
        let mut config = fixture(answer);
        let plain = config.requirements(WIDTH).unwrap();
        let policy = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
            max_pair_checks: 1_000_000, max_pair_bytes: 4_000_000 };
        config.encoding = UntrustedEncoding::SafeBpeV1(policy);
        let compact = config.requirements(WIDTH).unwrap();
        assert_eq!(compact.input_bytes - plain.input_bytes, policy.requirements().unwrap().workspace_bytes);
        assert_eq!(compact.preparation_bytes - plain.preparation_bytes, policy.requirements().unwrap().workspace_bytes);
        assert_eq!(compact.native, plain.native);
        assert_eq!(compact.output_reservation_bytes, plain.output_reservation_bytes);
        for threads in [2, 4, 64] {
            let team = config.requirements_with_threads(WIDTH, threads).unwrap();
            assert_eq!(team.input_bytes, compact.input_bytes);
            assert_eq!(team.workspace_bytes, compact.workspace_bytes);
            assert_eq!(team.team_workspace_bytes, int4_task_team_workspace_bytes(threads, WIDTH).unwrap());
            assert_eq!(team.total_bytes().unwrap(), compact.total_bytes().unwrap() + team.team_workspace_bytes);
        }
        for threads in [0, 1, 65, usize::MAX] {
            assert!(config.requirements_with_threads(WIDTH, threads).is_err());
        }
    }
}

#[test]
fn fixed_options_and_every_complete_run_bound_refuse_before_io() {
    for axis in 0..19 {
        let mut config = fixture(false);
        match axis {
            0 => config.work.max_model_work.forward_positions = config.options.max_model_work.forward_positions - 1,
            1 => config.work.max_model_work.projected_logits = config.options.max_model_work.projected_logits - 1,
            2 => config.work.max_model_work.attention_pairs = config.options.max_model_work.attention_pairs - 1,
            3 => config.work.max_model_work.projections.dot_products = config.options.max_model_work.projections.dot_products - 1,
            4 => config.work.max_model_work.projections.multiply_accumulates = config.options.max_model_work.projections.multiply_accumulates - 1,
            5 => config.work.masks.max_visits_per_run = config.work.masks.max_visits_per_item - 1,
            6 => config.work.masks.max_visits_per_item = 0,
            7 => {
                if let Int4SemanticSourceBatchTask::Summarize { budget, .. } = &mut config.task { budget.max_kv_bytes -= 1; }
            }
            8 => config.options.judge_budget.max_kv_bytes -= 1,
            9 => config.limits.preparation_reserve_bytes = u64::MAX,
            10 => config.limits.io_reserve_bytes = 0,
            11 => config.limits.transport.max_document_bytes = 0,
            12 => config.limits.transport.max_output_line_bytes = config.options.limits.max_output_bytes as usize - 1,
            13 => config.options.experimental_opt_in = false,
            14 => config.judge_planning.per_head.max_nodes = 0,
            15 => config.options.limits.max_statements = 2,
            16 => config.planning.max_passages = 0,
            17 => config.planning.max_context_tokens += 1,
            _ => config.limits.native.run.max_checkpoints = 1,
        }
        assert!(config.requirements(WIDTH).is_err(), "invalid source envelope axis {axis}");
    }
    let quote = fixture(false).requirements(WIDTH).unwrap();
    assert!(Int4SourceSemanticCorpusRequirements { input_bytes: u64::MAX, ..quote }.total_bytes().is_err());
}

#[test]
fn admission_binds_both_stages_full_result_and_qa_passage_geometry() {
    for answer in [false, true] {
        let config = fixture(answer);
        let source = identity(&config.task.built_in().spec().identity());
        let judge = identity("judge-v1");
        let required = config.requirements(WIDTH).unwrap();
        let admitted = request(&config, &source, &judge);
        assert!(admitted.max_result_bytes > admitted.source_budget.max_output_bytes);
        check_request(WIDTH, &config, required, &admitted).unwrap();
        for axis in 0..19 {
            let mut forged = request(&config, &source, &judge);
            match axis {
                0 => forged.group_width += 1,
                1 => forged.model_work.attention_pairs -= 1,
                2 => forged.source_reserved_work.forward_positions = 257,
                3 => forged.source_reserved_work.projections.dot_products = 0,
                4 => forged.mask_node_visits -= 1,
                5 => forged.mask_limits.checkpoint_interval_nodes += 1,
                6 => forged.kv_reservation_bytes -= 1,
                7 => forged.source_budget.max_output_bytes += 1,
                8 => forged.judge_budget.max_output_bytes += 1,
                9 => forged.semantic_limits.max_statements += 1,
                10 => forged.judge_planning_limits.max_total_candidate_tokens += 1,
                11 => forged.judge_planning_limits.per_head.max_nodes += 1,
                12 => forged.input_facts.source_bytes += 1,
                13 => forged.input_facts.source_bytes = 0,
                14 => forged.max_result_bytes -= 1,
                15 => forged.input_facts.question_bytes += 1,
                16 => forged.input_facts.passage_count += 1,
                17 => forged.input_facts.passage_count = if answer { 0 } else { 1 },
                _ => forged.max_result_bytes += 1,
            }
            assert!(check_request(WIDTH, &config, required, &forged).unwrap_err().stop, "binding {axis}");
        }
        for axis in 0..3 {
            let mut short = required;
            match axis { 0 => short.preparation_bytes -= 1, 1 => short.workspace_bytes -= 1,
                _ => short.output_reservation_bytes -= 1 }
            assert!(check_request(WIDTH, &config, short, &admitted).unwrap_err().stop);
        }
        let wrong = identity("sentiment-v1");
        assert!(check_request(WIDTH, &config, required, &request(&config, &wrong, &judge)).is_err());
        assert!(check_request(WIDTH, &config, required, &request(&config, &source, &wrong)).is_err());
    }
}

#[test]
fn invocation_owners_and_public_methods_cross_the_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceSemanticCorpusConfig>(); send::<StreamOwners>();
    send::<StreamExecution<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    send::<DeferredSourceSemanticStream<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    let _serial = NlpEngine::batch_int4_semantic_source::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int4_semantic_source_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let shown = format!("{:?}", fixture(true));
    assert!(shown.contains("answer-v1"));
    assert!(!shown.contains("judge_policy"));
}

mod lifecycle;
