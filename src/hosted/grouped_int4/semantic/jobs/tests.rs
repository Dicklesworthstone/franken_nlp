//! Pure admission and owned-runtime regression definitions, not model evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding, ExtractionMaskBudget},
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    jobs::population::PopulationReadLimits,
    native_engine::{constrained_int4::planned_work,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{extract::semantic::SemanticVerificationSpec, judge::FaithfulnessPolicy},
    tokenizer::untrusted::SafeBpeLimits,
};
use std::{io::Cursor, time::Duration};

const WIDTH: usize = 32;
const MIB: u64 = 1 << 20;

fn maximum() -> Int4Work {
    Int4Work { forward_positions: u64::MAX / 8, projected_logits: u64::MAX / 8,
        attention_pairs: u64::MAX / 8, projections: ProjectionWork {
            dot_products: u64::MAX / 8, multiply_accumulates: u64::MAX / 8 } }
}
fn fixture() -> (Int4SemanticExtractionJobConfig, JobLimits, JobHostLimits) {
    let native = Int4MemoryRequirement::for_context(256, WIDTH).unwrap();
    let extraction = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
    let options = Int4SemanticOptions {
        specification: SemanticVerificationSpec { schema_version: 1, experimental_opt_in: true,
            revision: "private-review-policy".to_owned(), claims: vec![], judge_budget: judge,
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0,
                minimum_margin_milli: 0, evidence_window_bytes: 4096,
                max_evidence_windows: 8, max_evidence_spans: 8 } },
        limits: SemanticLimits { max_fields: 3, max_claim_bytes: 128,
            max_total_claim_bytes: 1024, max_template_bytes: 1024, max_walk_nodes: 128,
            max_input_json_bytes: 8192, max_total_prompt_tokens: 2048,
            max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 },
        max_model_work: maximum(),
    };
    let work = Int4SemanticExtractionBatchLimits { max_model_work: maximum().checked_add(maximum()).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
            max_visits_per_item: 32_000, max_visits_per_run: 128_000 } };
    let config = Int4SemanticExtractionJobConfig {
        extraction: ExtractionBatchArgs { schema: r#"{"type":"number"}"#.to_owned(),
            grounding: ExtractionBatchGrounding::Structural, budget: extraction },
        compiler_limits: CompileLimits::default(), source_limits: SourceRuntimeLimits::default(),
        options, judge_planning: JudgeLimits { max_total_prompt_tokens: 2048,
            max_total_candidate_tokens: 256, max_total_projected_logits: 1_000_000,
            max_output_bytes: 16_384, ..JudgeLimits::default() },
        encoding: UntrustedEncoding::ByteFallback, work,
    };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 131_072, max_spool_bytes: MIB,
        max_materialized_bytes: MIB, max_journal_bytes: MIB, max_attempts: 4,
        max_work: JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 1_000_000,
                cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: MIB, max_lines: 100 },
        preparation_reserve_bytes: 512 * MIB, io_reserve_bytes: 131_072,
        journal_reserve_bytes: MIB, serialization_reserve_bytes: MIB,
    };
    (config, job, host)
}
fn quote(config: &Int4SemanticExtractionJobConfig, job: JobLimits, host: JobHostLimits,
    input: JobInputMemory, threads: Option<usize>) -> Result<Int4SemanticJobRequirements, HostedJobError> {
    int4_semantic_job_requirements(WIDTH, config.extraction.budget, &config.options,
        config.judge_planning, &config.encoding, config.work, host, job, input, threads)
}

#[test]
fn complete_semantic_preparation_replaces_only_the_shared_job_preparation_component() {
    let (config, job, host) = fixture();
    let semantic = semantic_host(host, config.work).requirements_with_encoding(WIDTH,
        config.extraction.budget, config.options.specification.judge_budget,
        config.options.limits, config.judge_planning, job.max_input_bytes_per_item,
        &config.encoding).unwrap();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let required = quote(&config, job, host, input, None).unwrap();
        assert_eq!(required.native, semantic.native);
        assert_eq!(required.workspace_bytes, semantic.native_workspace_bytes);
        assert_eq!(required.preparation_bytes, semantic.preparation_bytes
            + 2 * SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64);
        assert_eq!(required.output_reservation_bytes, semantic.output_bytes);
        assert_eq!(required.input_bytes, host.required_buffer_bytes_for_input(job, input).unwrap()
            - host.preparation_reserve_bytes + semantic.preparation_bytes
            + 2 * SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64);
        assert_eq!(required.team_workspace_bytes, 0);
        assert_eq!(required.total_bytes().unwrap(), required.native.kv_bytes + required.workspace_bytes
            + required.input_bytes + required.output_reservation_bytes);
        assert!(required.output_reservation_bytes >= 4 * config.options.limits.max_output_bytes);
    }
    let complete = quote(&config, job, host, JobInputMemory::CompletePopulation, None).unwrap();
    let retained = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    assert_eq!(complete.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(complete.native, retained.native);
    assert_eq!(complete.preparation_bytes, retained.preparation_bytes);
    assert_eq!(complete.output_reservation_bytes, retained.output_reservation_bytes);
}

#[test]
fn recipe_buffers_are_a_checked_floor_independent_of_caller_preparation() {
    let (config, job, mut host) = fixture();
    host.preparation_reserve_bytes = 1;
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let semantic = semantic_host(host, config.work).requirements_with_encoding(WIDTH,
            config.extraction.budget, config.options.specification.judge_budget,
            config.options.limits, config.judge_planning, job.max_input_bytes_per_item,
            &config.encoding).unwrap();
        let required = quote(&config, job, host, input, None).unwrap();
        assert_eq!(required.preparation_bytes - semantic.preparation_bytes,
            2 * SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64);
        assert!(required.input_bytes >= required.preparation_bytes);
    }
    assert!(preparation_with_recipe(u64::MAX).is_err());
    let floor = 2 * SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64;
    assert_eq!(preparation_with_recipe(u64::MAX - floor).unwrap(), u64::MAX);
    assert!(preparation_with_recipe(u64::MAX - floor + 1).is_err());
}

#[test]
fn encoder_and_persistent_team_are_separate_peak_owners_and_never_multiply_per_field() {
    let (mut config, job, host) = fixture();
    let encoding = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    for fields in [1, 3, 64] {
        config.options.limits.max_fields = fields;
        config.encoding = UntrustedEncoding::ByteFallback;
        let base = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
        config.encoding = UntrustedEncoding::SafeBpeV1(encoding);
        let compact = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
        let workspace = encoding.requirements().unwrap().workspace_bytes;
        assert_eq!(compact.preparation_bytes - base.preparation_bytes, workspace);
        assert_eq!(compact.input_bytes - base.input_bytes, workspace);
        assert_eq!(compact.native, base.native);
        assert_eq!(compact.workspace_bytes, base.workspace_bytes);
        assert_eq!(compact.output_reservation_bytes, base.output_reservation_bytes);
        for threads in [2, 4, 64] {
            let team = quote(&config, job, host, JobInputMemory::RetainedRecord, Some(threads)).unwrap();
            let extra = int4_task_team_workspace_bytes(threads, WIDTH).unwrap();
            assert_eq!(team.team_workspace_bytes, extra);
            assert_eq!(team.input_bytes, compact.input_bytes);
            assert_eq!(team.total_bytes().unwrap() - compact.total_bytes().unwrap(), extra);
        }
    }
}

#[test]
fn every_lifetime_axis_and_full_attempt_must_fit_before_private_io() {
    let (config, job, host) = fixture();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        for axis in 0..6 {
            let mut changed = job;
            match axis {
                0 => changed.max_work.model.forward_positions += 1,
                1 => changed.max_work.model.projected_logits += 1,
                2 => changed.max_work.model.attention_pairs += 1,
                3 => changed.max_work.model.projections.dot_products += 1,
                4 => changed.max_work.model.projections.multiply_accumulates += 1,
                _ => changed.max_work.mask_node_visits += 1,
            }
            assert!(quote(&config, changed, host, input, None).is_err(), "axis {axis}");
        }
        let mut too_small = job; too_small.max_result_bytes -= 1;
        assert!(quote(&config, too_small, host, input, None).is_err());
        let mut too_large = config.clone();
        too_large.options.max_model_work.forward_positions = config.work.max_model_work.forward_positions + 1;
        assert!(quote(&too_large, job, host, input, None).is_err());
    }
}

#[test]
fn missing_reserves_invalid_judges_and_arithmetic_overflow_refuse() {
    let (config, job, host) = fixture();
    for axis in 0..9 {
        let mut altered = host;
        match axis {
            0 => altered.preparation_reserve_bytes = 0,
            1 => altered.io_reserve_bytes = 0,
            2 => altered.journal_reserve_bytes = 0,
            3 => altered.serialization_reserve_bytes = 0,
            4 => altered.native.allocator_reserve_bytes = 0,
            5 => altered.preparation_reserve_bytes = u64::MAX,
            6 => altered.io_reserve_bytes = u64::MAX,
            7 => altered.journal_reserve_bytes = u64::MAX,
            _ => altered.serialization_reserve_bytes = u64::MAX,
        }
        assert!(quote(&config, job, altered, JobInputMemory::RetainedRecord, None).is_err(), "axis {axis}");
    }
    let mut invalid = config.clone(); invalid.judge_planning.per_head.max_nodes = 0;
    assert!(quote(&invalid, job, host, JobInputMemory::RetainedRecord, None).is_err());
    for threads in [0, 65, usize::MAX] {
        assert!(quote(&config, job, host, JobInputMemory::RetainedRecord, Some(threads)).is_err());
    }
    let required = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    assert!(Int4SemanticJobRequirements { output_reservation_bytes: u64::MAX, ..required }.total_bytes().is_err());
}

fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"semantic-job-admission-fixture");
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
#[test]
fn dynamic_source_and_combined_review_cannot_exceed_the_admitted_job_owner() {
    let (config, job, host) = fixture();
    let required = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    let extraction = identity("extract-v1"); let judge = identity("judge-v1");
    let make = || Int4SemanticExtractionAdmission {
        identity: &extraction, judge_identity: &judge, group_width: WIDTH,
        model_work: config.options.max_model_work,
        extraction_reserved_work: planned_work(WIDTH, 64, 8).unwrap(),
        mask_node_visits: config.work.masks.max_visits_per_item,
        mask_limits: config.work.masks.per_mask, kv_reservation_bytes: required.native.kv_bytes,
        max_result_bytes: config.options.limits.max_output_bytes,
        extraction_budget: config.extraction.budget, judge_budget: config.options.specification.judge_budget,
        semantic_limits: config.options.limits, judge_planning_limits: config.judge_planning,
        source_bytes: job.max_input_bytes_per_item, input_encoding: &config.encoding,
    };
    check_request(WIDTH, &config, host, job, required, &make()).unwrap();
    for axis in 0..10 {
        let mut request = make();
        match axis {
            0 => request.source_bytes += 1,
            1 => request.model_work.forward_positions -= 1,
            2 => request.mask_node_visits += 1,
            3 => request.mask_limits.checkpoint_interval_nodes += 1,
            4 => request.kv_reservation_bytes -= 1,
            5 => request.max_result_bytes -= 1,
            6 => request.extraction_budget.max_output_tokens -= 1,
            7 => request.judge_budget.max_input_tokens -= 1,
            8 => request.semantic_limits.max_fields += 1,
            _ => request.judge_planning_limits.per_head.max_nodes += 1,
        }
        assert!(check_request(WIDTH, &config, host, job, required, &request).is_err(), "axis {axis}");
    }
    let short = Int4SemanticJobRequirements { preparation_bytes: required.preparation_bytes - 1, ..required };
    assert!(check_request(WIDTH, &config, host, job, short, &make()).is_err());
}

#[test]
fn private_inputs_and_complete_output_claims_cross_the_owned_job_boundary() {
    fn send<T: Send + 'static>() {}
    send::<JobInput<Cursor<Vec<u8>>>>();
    send::<JobExecution<Cursor<Vec<u8>>>>();
    send::<DeferredSemanticJob<Cursor<Vec<u8>>>>();
    send::<Arc<JobOwners>>();
    send::<Arc<Int4SemanticExtractionBatchPlanner>>();
    send::<JobPopulation>();
    let _serial = NlpEngine::job_int4_semantic_extract::<Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_semantic_extract_with_threads::<Cursor<Vec<u8>>>;
}

mod lifecycle;
mod model;
