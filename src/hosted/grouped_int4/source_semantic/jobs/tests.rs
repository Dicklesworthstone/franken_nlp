//! Pure complete-source admission and owned-runtime regression definitions.
//! These fixtures do not supply inference or semantic-quality evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{extract::ExtractionMaskBudget, source::semantic::Int4SemanticSourceBatchTask},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    jobs::population::PopulationReadLimits,
    native_engine::{constrained_int4::planned_work,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{answer::AnswerOptions, judge::FaithfulnessPolicy, summarize::SummaryOptions},
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
fn fixture(answer: bool) -> (Int4SemanticSourceJobConfig, JobLimits, JobHostLimits) {
    let native = Int4MemoryRequirement::for_context(256, WIDTH).unwrap();
    let source = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
    let task = if answer {
        Int4SemanticSourceBatchTask::Answer { options: AnswerOptions::default(), budget: source }
    } else {
        Int4SemanticSourceBatchTask::Summarize { options: SummaryOptions {
            max_bullets: 3, max_bullet_scalars: 32, max_citations_per_bullet: 2, max_quote_scalars: 32,
        }, budget: source }
    };
    let options = Int4SourceSemanticOptions { schema_version: 1, experimental_opt_in: true,
        judge_budget: judge, judge_policy: FaithfulnessPolicy {
            minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0, evidence_window_bytes: 4096,
            max_evidence_windows: 8, max_evidence_spans: 8 },
        limits: SourceSemanticLimits { max_statements: 3, max_source_bytes: 32_768,
            max_claim_bytes: 128, max_total_claim_bytes: 1024, max_total_prompt_tokens: 2048,
            max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 },
        max_model_work: maximum(),
    };
    let work = Int4SemanticSourceBatchLimits { max_model_work: maximum().checked_add(maximum()).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
            max_visits_per_item: 32_000, max_visits_per_run: 128_000 } };
    let config = Int4SemanticSourceJobConfig {
        task, planning: SourcePlanningLimits { max_input_bytes: 1024,
            max_context_tokens: 256, max_passages: 4, ..SourcePlanningLimits::default() },
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
fn quote(config: &Int4SemanticSourceJobConfig, job: JobLimits, host: JobHostLimits,
    input: JobInputMemory, threads: Option<usize>) -> Result<Int4SourceSemanticJobRequirements, HostedJobError> {
    int4_source_semantic_job_requirements(WIDTH, config.task.budget(), &config.options,
        config.judge_planning, input_ceiling(config, host, job)?, &config.encoding,
        config.work, host, job, input, threads)
}
fn base(config: &Int4SemanticSourceJobConfig, job: JobLimits, host: JobHostLimits)
    -> Int4SourceSemanticHostRequirements {
    semantic_host(host, config.work).requirements_with_encoding(WIDTH, config.task.budget(),
        config.options.judge_budget, config.options.limits, config.judge_planning,
        input_ceiling(config, host, job).unwrap(), &config.encoding).unwrap()
}

#[test]
fn complete_source_reviews_replace_only_the_shared_job_preparation_component() {
    for answer in [false, true] {
        let (config, job, host) = fixture(answer);
        let semantic = base(&config, job, host);
        for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
            let required = quote(&config, job, host, input, None).unwrap();
            assert_eq!(required.native, semantic.native);
            assert_eq!(required.workspace_bytes, semantic.native_workspace_bytes);
            assert_eq!(required.preparation_bytes, semantic.preparation_bytes
                + 2 * SEMANTIC_SOURCE_JOB_RECIPE_BYTES as u64);
            assert_eq!(required.output_reservation_bytes, semantic.output_bytes);
            assert_eq!(required.input_bytes, host.required_buffer_bytes_for_input(job, input).unwrap()
                - host.preparation_reserve_bytes + required.preparation_bytes);
            assert_eq!(required.team_workspace_bytes, 0);
            assert_eq!(required.total_bytes().unwrap(), required.native.kv_bytes + required.workspace_bytes
                + required.input_bytes + required.output_reservation_bytes);
            assert!(required.output_reservation_bytes > 4 * config.options.limits.max_output_bytes);
        }
        let complete = quote(&config, job, host, JobInputMemory::CompletePopulation, None).unwrap();
        let retained = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
        assert_eq!(complete.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
        assert_eq!(complete.native, retained.native);
        assert_eq!(complete.preparation_bytes, retained.preparation_bytes);
        assert_eq!(complete.output_reservation_bytes, retained.output_reservation_bytes);
    }
}

#[test]
fn questions_and_every_passage_are_priced_beyond_the_full_source_with_an_exact_record_ceiling() {
    let (summary, job, host) = fixture(false);
    let summary = quote(&summary, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    let (mut config, _, _) = fixture(true);
    let qa = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    assert_eq!(qa.preparation_bytes - summary.preparation_bytes, 8 * 128 + 4 * 2048);
    config.options.limits.max_claim_bytes *= 2;
    config.planning.max_passages += 1;
    let expanded = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    assert_eq!(expanded.preparation_bytes - qa.preparation_bytes, 8 * 128 + 2048);
    config.options.limits.max_source_bytes = 512;
    assert_eq!(input_ceiling(&config, host, job).unwrap().source_bytes, 512);
    config.planning.max_input_bytes = job.max_input_bytes_per_item + 1;
    assert!(quote(&config, job, host, JobInputMemory::RetainedRecord, None).is_err());
    assert_eq!(config.planning.max_input_bytes, job.max_input_bytes_per_item + 1,
        "preflight cannot silently mutate the authenticated configuration");
    config.planning.max_input_bytes = job.max_input_bytes_per_item;
    let facts = input_ceiling(&config, host, job).unwrap();
    let mut too_large = facts; too_large.source_bytes = job.max_input_bytes_per_item + 1;
    assert!(int4_source_semantic_job_requirements(WIDTH, config.task.budget(), &config.options,
        config.judge_planning, too_large, &config.encoding, config.work, host, job,
        JobInputMemory::RetainedRecord, None).is_err());
    let mut too_large = facts; too_large.question_bytes = job.max_input_bytes_per_item + 1;
    assert!(int4_source_semantic_job_requirements(WIDTH, config.task.budget(), &config.options,
        config.judge_planning, too_large, &config.encoding, config.work, host, job,
        JobInputMemory::RetainedRecord, None).is_err());
}

#[test]
fn both_recipe_buffers_are_checked_floors_independent_of_caller_preparation() {
    let (config, job, mut host) = fixture(false); host.preparation_reserve_bytes = 1;
    let semantic = base(&config, job, host);
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let required = quote(&config, job, host, input, None).unwrap();
        assert_eq!(required.preparation_bytes - semantic.preparation_bytes,
            2 * SEMANTIC_SOURCE_JOB_RECIPE_BYTES as u64);
        assert!(required.input_bytes >= required.preparation_bytes);
    }
    let floor = 2 * SEMANTIC_SOURCE_JOB_RECIPE_BYTES as u64;
    assert!(preparation_with_recipe(u64::MAX).is_err());
    assert_eq!(preparation_with_recipe(u64::MAX - floor).unwrap(), u64::MAX);
    assert!(preparation_with_recipe(u64::MAX - floor + 1).is_err());
}

#[test]
fn encoder_and_persistent_team_have_single_independent_owners_for_all_statement_ceilings() {
    let encoding = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    for answer in [false, true] { for statements in [3, 64] {
        let (mut config, job, host) = fixture(answer);
        config.options.limits.max_statements = statements;
        let plain = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
        config.encoding = UntrustedEncoding::SafeBpeV1(encoding);
        let compact = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
        let workspace = encoding.requirements().unwrap().workspace_bytes;
        assert_eq!(compact.preparation_bytes - plain.preparation_bytes, workspace);
        assert_eq!(compact.input_bytes - plain.input_bytes, workspace);
        assert_eq!(compact.native, plain.native);
        assert_eq!(compact.workspace_bytes, plain.workspace_bytes);
        assert_eq!(compact.output_reservation_bytes, plain.output_reservation_bytes);
        for threads in [2, 4, 64] {
            let team = quote(&config, job, host, JobInputMemory::RetainedRecord, Some(threads)).unwrap();
            let extra = int4_task_team_workspace_bytes(threads, WIDTH).unwrap();
            assert_eq!(team.team_workspace_bytes, extra);
            assert_eq!(team.input_bytes, compact.input_bytes);
            assert_eq!(team.total_bytes().unwrap() - compact.total_bytes().unwrap(), extra);
        }
    } }
}

#[test]
fn every_lifetime_axis_full_attempt_and_complete_result_must_fit_before_private_io() {
    for answer in [false, true] {
        let (config, job, host) = fixture(answer);
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
                assert!(quote(&config, changed, host, input, None).is_err(), "lifetime axis {axis}");
            }
            let mut too_small = job; too_small.max_result_bytes -= 1;
            assert!(quote(&config, too_small, host, input, None).is_err());
            let mut too_wide = job; too_wide.max_id_bytes = 257;
            assert!(quote(&config, too_wide, host, input, None).is_err());
            for axis in 0..5 {
                let mut changed = config.clone();
                match axis {
                    0 => changed.options.max_model_work.forward_positions = config.work.max_model_work.forward_positions + 1,
                    1 => changed.options.max_model_work.projected_logits = config.work.max_model_work.projected_logits + 1,
                    2 => changed.options.max_model_work.attention_pairs = config.work.max_model_work.attention_pairs + 1,
                    3 => changed.options.max_model_work.projections.dot_products = config.work.max_model_work.projections.dot_products + 1,
                    _ => changed.options.max_model_work.projections.multiply_accumulates = config.work.max_model_work.projections.multiply_accumulates + 1,
                }
                assert!(quote(&changed, job, host, input, None).is_err(), "full attempt axis {axis}");
            }
        }
    }
}

#[test]
fn missing_reserves_both_resident_kvs_invalid_judges_and_arithmetic_overflow_refuse() {
    let (config, job, host) = fixture(false);
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
        assert!(quote(&config, job, altered, JobInputMemory::RetainedRecord, None).is_err(), "reserve {axis}");
    }
    for stage in 0..2 {
        let mut invalid = config.clone();
        if stage == 0 {
            if let Int4SemanticSourceBatchTask::Summarize { budget, .. } = &mut invalid.task {
                budget.max_kv_bytes -= 1;
            }
        } else { invalid.options.judge_budget.max_kv_bytes -= 1; }
        assert!(quote(&invalid, job, host, JobInputMemory::RetainedRecord, None).is_err());
    }
    let mut invalid = config.clone(); invalid.judge_planning.per_head.max_nodes = 0;
    assert!(quote(&invalid, job, host, JobInputMemory::RetainedRecord, None).is_err());
    let mut invalid = config.clone(); invalid.options.limits.max_statements = 2;
    assert!(quote(&invalid, job, host, JobInputMemory::RetainedRecord, None).is_err());
    for threads in [0, 1, 65, usize::MAX] {
        assert!(quote(&config, job, host, JobInputMemory::RetainedRecord, Some(threads)).is_err());
    }
    let required = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    assert!(Int4SourceSemanticJobRequirements { output_reservation_bytes: u64::MAX, ..required }.total_bytes().is_err());
}

fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"semantic-source-job-admission-fixture");
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
fn dynamic_source_and_combined_review_cannot_exceed_or_change_the_admitted_job_owner() {
    for answer in [false, true] {
        let (config, job, host) = fixture(answer);
        let required = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
        let source = identity(&config.task.built_in().spec().identity()); let judge = identity("judge-v1");
        let make = || Int4SemanticSourceAdmission {
            identity: &source, judge_identity: &judge, group_width: WIDTH,
            model_work: config.options.max_model_work,
            source_reserved_work: planned_work(WIDTH, 64, 8).unwrap(),
            mask_node_visits: config.work.masks.max_visits_per_item,
            mask_limits: config.work.masks.per_mask, kv_reservation_bytes: required.native.kv_bytes,
            max_result_bytes: config.options.limits.max_output_bytes,
            source_budget: config.task.budget(), judge_budget: config.options.judge_budget,
            semantic_limits: config.options.limits, judge_planning_limits: config.judge_planning,
            input_facts: input_ceiling(&config, host, job).unwrap(), input_encoding: &config.encoding,
        };
        check_request(WIDTH, &config, host, job, required, &make()).unwrap();
        for axis in 0..19 {
            let mut request = make();
            match axis {
                0 => request.group_width += 1,
                1 => request.model_work.attention_pairs -= 1,
                2 => request.source_reserved_work.forward_positions = 0,
                3 => request.source_reserved_work.projected_logits = 0,
                4 => request.source_reserved_work.attention_pairs = 0,
                5 => request.source_reserved_work.projections.dot_products = 0,
                6 => request.source_reserved_work.projections.multiply_accumulates = 0,
                7 => request.mask_node_visits += 1,
                8 => request.mask_limits.checkpoint_interval_nodes += 1,
                9 => request.kv_reservation_bytes -= 1,
                10 => request.max_result_bytes -= 1,
                11 => request.source_budget.max_output_tokens -= 1,
                12 => request.judge_budget.max_input_tokens -= 1,
                13 => request.semantic_limits.max_statements += 1,
                14 => request.judge_planning_limits.per_head.max_nodes += 1,
                15 => request.input_facts.source_bytes += 1,
                16 => request.input_facts.question_bytes += 1,
                17 => request.input_facts.passage_count += 1,
                _ => request.input_facts.passage_count = if answer { 0 } else { 1 },
            }
            assert!(check_request(WIDTH, &config, host, job, required, &request).unwrap_err().stop,
                "admission axis {axis}");
        }
        for axis in 0..3 {
            let mut short = required;
            match axis { 0 => short.preparation_bytes -= 1, 1 => short.workspace_bytes -= 1,
                _ => short.output_reservation_bytes -= 1 }
            assert!(check_request(WIDTH, &config, host, job, short, &make()).is_err());
        }
        let wrong = identity("sentiment-v1");
        let mut request = make(); request.identity = &wrong;
        assert!(check_request(WIDTH, &config, host, job, required, &request).is_err());
        let mut request = make(); request.judge_identity = &wrong;
        assert!(check_request(WIDTH, &config, host, job, required, &request).is_err());
        let other_encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 8192, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
        let mut request = make(); request.input_encoding = &other_encoding;
        assert!(check_request(WIDTH, &config, host, job, required, &request).is_err());
    }
}

#[test]
fn actual_shared_compilers_private_inputs_and_complete_results_cross_the_owned_job_boundary() {
    fn send<T: Send + 'static>() {}
    send::<JobInput<Cursor<Vec<u8>>>>();
    send::<JobExecution<Cursor<Vec<u8>>>>();
    send::<DeferredSourceSemanticJob<Cursor<Vec<u8>>>>();
    send::<Arc<JobOwners>>();
    send::<Arc<Int4SemanticSourceBatchPlanner<'static>>>();
    send::<JobPopulation>();
    let _serial = NlpEngine::job_int4_semantic_source::<Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_semantic_source_with_threads::<Cursor<Vec<u8>>>;
}

mod lifecycle;
mod model;
