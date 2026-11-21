//! Pure admission and privacy fixtures; no model-quality evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchLimits, BatchWork, extract::{ExtractionBatchGrounding, ExtractionMaskBudget}},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{constrained_int4::planned_work, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::extract::semantic::SemanticVerificationSpec,
    tasks::judge::FaithfulnessPolicy,
    tokenizer::untrusted::SafeBpeLimits,
};

const WIDTH: usize = 32;
const MIB: u64 = 1 << 20;

fn work() -> Int4Work {
    Int4Work { forward_positions: u64::MAX / 8, projected_logits: u64::MAX / 8,
        attention_pairs: u64::MAX / 8, projections: ProjectionWork {
            dot_products: u64::MAX / 8, multiply_accumulates: u64::MAX / 8 } }
}
fn fixture() -> Int4SemanticCorpusConfig {
    let kv = Int4MemoryRequirement::for_context(256, WIDTH).unwrap().kv_bytes;
    let extraction = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: kv };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 4096, max_kv_bytes: kv };
    let semantic = SemanticLimits { max_fields: 3, max_claim_bytes: 128,
        max_total_claim_bytes: 1024, max_template_bytes: 1024, max_walk_nodes: 128,
        max_input_json_bytes: 8192, max_total_prompt_tokens: 2048,
        max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 };
    let run_work = work().checked_add(work()).unwrap();
    Int4SemanticCorpusConfig {
        extraction: ExtractionBatchArgs { schema: r#"{"type":"number"}"#.to_owned(),
            grounding: ExtractionBatchGrounding::Structural, budget: extraction },
        compiler_limits: CompileLimits::default(), source_limits: SourceRuntimeLimits::default(),
        options: Int4SemanticOptions { specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "private-review-revision".to_owned(),
            claims: vec![], judge_budget: judge, judge_policy: FaithfulnessPolicy {
                minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0, evidence_window_bytes: 4096,
                max_evidence_windows: 8, max_evidence_spans: 8 },
        }, limits: semantic, max_model_work: work() },
        judge_planning: JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
            max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() },
        encoding: UntrustedEncoding::ByteFallback,
        work: Int4SemanticExtractionBatchLimits { max_model_work: run_work,
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
fn base(config: &Int4SemanticCorpusConfig) -> Int4SemanticHostRequirements {
    semantic_host(config.work, config.limits).requirements_with_encoding(WIDTH, config.extraction.budget,
        config.options.specification.judge_budget, config.options.limits, config.judge_planning,
        config.limits.transport.max_document_bytes, &config.encoding).unwrap()
}
fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"semantic-corpus-admission-fixture");
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
fn request<'a>(config: &'a Int4SemanticCorpusConfig, extraction: &'a ExecutionIdentity,
    judge: &'a ExecutionIdentity) -> Int4SemanticExtractionAdmission<'a> {
    let required = config.requirements(WIDTH).unwrap();
    Int4SemanticExtractionAdmission {
        identity: extraction, judge_identity: judge, group_width: WIDTH,
        model_work: config.options.max_model_work,
        extraction_reserved_work: planned_work(WIDTH, 64, 8).unwrap(),
        mask_node_visits: config.work.masks.max_visits_per_item,
        mask_limits: config.work.masks.per_mask, kv_reservation_bytes: required.native.kv_bytes,
        max_result_bytes: config.options.limits.max_output_bytes, extraction_budget: config.extraction.budget,
        judge_budget: config.options.specification.judge_budget, semantic_limits: config.options.limits,
        judge_planning_limits: config.judge_planning, source_bytes: config.limits.transport.max_document_bytes,
        input_encoding: &config.encoding,
    }
}

#[test]
fn full_semantic_peak_adds_transport_once_without_discounting_review_intermediates() {
    let config = fixture();
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
}

#[test]
fn larger_runs_and_total_input_do_not_multiply_the_document_source_or_model_geometry() {
    let mut config = fixture();
    let before = config.requirements(WIDTH).unwrap();
    config.work.max_model_work = config.work.max_model_work.checked_add(config.work.max_model_work).unwrap();
    config.work.masks.max_visits_per_run *= 2;
    config.limits.transport.max_requests *= 2;
    config.limits.transport.max_input_bytes *= 2;
    assert_eq!(config.requirements(WIDTH).unwrap(), before);
    config.limits.transport.max_document_bytes *= 2;
    let larger_source = config.requirements(WIDTH).unwrap();
    assert_eq!(larger_source.native, before.native);
    assert_eq!(larger_source.workspace_bytes, before.workspace_bytes);
    assert_eq!(larger_source.output_reservation_bytes, before.output_reservation_bytes);
    assert_eq!(larger_source.preparation_bytes - before.preparation_bytes,
        (config.options.limits.max_fields as u64 + 2) * 1024);
}

#[test]
fn compact_encoding_and_fixed_workers_are_independent_once_per_invocation() {
    let mut config = fixture();
    let plain = config.requirements(WIDTH).unwrap();
    let policy = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1_000_000, max_pair_bytes: 4_000_000 };
    config.encoding = UntrustedEncoding::SafeBpeV1(policy);
    let compact = config.requirements(WIDTH).unwrap();
    assert_eq!(compact.input_bytes - plain.input_bytes, policy.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.preparation_bytes - plain.preparation_bytes, policy.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, plain.native);
    assert_eq!(compact.workspace_bytes, plain.workspace_bytes);
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

#[test]
fn invalid_complete_document_and_run_envelopes_refuse_before_io() {
    for axis in 0..17 {
        let mut config = fixture();
        match axis {
            0 => config.work.max_model_work.forward_positions = config.options.max_model_work.forward_positions - 1,
            1 => config.work.max_model_work.projected_logits = config.options.max_model_work.projected_logits - 1,
            2 => config.work.max_model_work.attention_pairs = config.options.max_model_work.attention_pairs - 1,
            3 => config.work.max_model_work.projections.dot_products = config.options.max_model_work.projections.dot_products - 1,
            4 => config.work.max_model_work.projections.multiply_accumulates = config.options.max_model_work.projections.multiply_accumulates - 1,
            5 => config.work.masks.max_visits_per_run = config.work.masks.max_visits_per_item - 1,
            6 => config.work.masks.max_visits_per_item = 0,
            7 => config.extraction.budget.max_kv_bytes -= 1,
            8 => config.options.specification.judge_budget.max_kv_bytes -= 1,
            9 => config.limits.preparation_reserve_bytes = u64::MAX,
            10 => config.limits.io_reserve_bytes = 0,
            11 => config.limits.transport.max_document_bytes = 0,
            12 => config.limits.transport.max_output_line_bytes = config.options.limits.max_output_bytes as usize - 1,
            13 => config.options.specification.experimental_opt_in = false,
            14 => config.judge_planning.per_head.max_nodes = 0,
            15 => config.extraction.schema.clear(),
            _ => config.limits.native.run.max_checkpoints = 1,
        }
        assert!(config.requirements(WIDTH).is_err(), "invalid envelope axis {axis}");
    }
    let quote = fixture().requirements(WIDTH).unwrap();
    assert!(Int4SemanticCorpusRequirements { input_bytes: u64::MAX, ..quote }.total_bytes().is_err());
}

#[test]
fn admission_binds_both_stages_and_the_full_output_instead_of_the_extraction_output() {
    let config = fixture();
    let extraction = identity("extract-v1"); let judge = identity("judge-v1");
    let required = config.requirements(WIDTH).unwrap();
    let admitted = request(&config, &extraction, &judge);
    assert!(admitted.max_result_bytes > admitted.extraction_budget.max_output_bytes);
    check_request(WIDTH, &config, required, &admitted).unwrap();
    for axis in 0..16 {
        let mut forged = request(&config, &extraction, &judge);
        match axis {
            0 => forged.group_width += 1,
            1 => forged.model_work.attention_pairs -= 1,
            2 => forged.extraction_reserved_work.forward_positions = 257,
            3 => forged.extraction_reserved_work.projections.dot_products = 0,
            4 => forged.mask_node_visits -= 1,
            5 => forged.mask_limits.checkpoint_interval_nodes += 1,
            6 => forged.kv_reservation_bytes -= 1,
            7 => forged.extraction_budget.max_output_bytes += 1,
            8 => forged.judge_budget.max_output_bytes += 1,
            9 => forged.semantic_limits.max_fields += 1,
            10 => forged.judge_planning_limits.max_total_candidate_tokens += 1,
            11 => forged.judge_planning_limits.per_head.max_nodes += 1,
            12 => forged.source_bytes += 1,
            13 => forged.source_bytes = 0,
            14 => forged.max_result_bytes -= 1,
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
    assert!(check_request(WIDTH, &config, required, &request(&config, &extraction, &wrong)).is_err());
}

#[test]
fn invocation_owners_cross_the_blocking_boundary_without_debugging_private_configuration() {
    fn send<T: Send + 'static>() {}
    send::<Int4SemanticCorpusConfig>(); send::<StreamOwners>();
    send::<StreamExecution<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    send::<DeferredSemanticStream<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    let _serial = NlpEngine::batch_int4_semantic_extract::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int4_semantic_extract_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let shown = format!("{:?}", fixture());
    assert!(!shown.contains("private-review-revision"));
    assert!(!shown.contains(r#"{"type":"number"}"#));
}

mod lifecycle;
