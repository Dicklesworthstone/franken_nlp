//! Pure durable admission definitions; model and crash qualification stay separate.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding, ExtractionMaskBudget},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    jobs::runner::extract::semantic::Int8SemanticExtractionJobConfig,
    native_engine::portable_int8::ProjectionWork,
    tasks::{extract::semantic::{SemanticLimits, SemanticVerificationSpec}, judge::FaithfulnessPolicy},
    tokenizer::untrusted::SafeBpeLimits,
};

fn fixture() -> (Int8SemanticExtractionJobConfig, JobLimits, JobHostLimits) {
    let kv = Int8MemoryRequirement::for_context(256).unwrap().kv_bytes;
    let extraction = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: kv };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 4096, max_kv_bytes: kv };
    let document = Int8Work { forward_positions: 100_000, projected_logits: 1_000_000_000,
        attention_pairs: 10_000_000_000, projections: ProjectionWork {
            dot_products: 100_000_000_000, multiply_accumulates: 100_000_000_000_000 } };
    let work = Int8SemanticExtractionBatchLimits { max_model_work: document.checked_add(document).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
            max_visits_per_item: 32_000, max_visits_per_run: 128_000 } };
    let config = Int8SemanticExtractionJobConfig {
        extraction: ExtractionBatchArgs { schema: r#"{"type":"number"}"#.into(),
            grounding: ExtractionBatchGrounding::Structural, budget: extraction },
        compiler_limits: CompileLimits::default(), source_limits: SourceRuntimeLimits::default(),
        options: Int8SemanticOptions {
            specification: SemanticVerificationSpec { schema_version: 1, experimental_opt_in: true,
                revision: "semantic-job-fixture".into(), claims: vec![], judge_budget: judge,
                judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0,
                    minimum_margin_milli: 0, evidence_window_bytes: 4096,
                    max_evidence_windows: 8, max_evidence_spans: 8 } },
            limits: SemanticLimits { max_fields: 3, max_claim_bytes: 128,
                max_total_claim_bytes: 1024, max_template_bytes: 1024, max_walk_nodes: 128,
                max_input_json_bytes: 8192, max_total_prompt_tokens: 2048,
                max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 },
            max_model_work: document,
        },
        judge_planning: JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
            max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() },
        encoding: UntrustedEncoding::ByteFallback, encoding_work_limits: None, work,
    };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 131_072, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: work.max_model_work, mask_node_visits: work.masks.max_visits_per_run } };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 1_000_000,
                cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 512 << 20, io_reserve_bytes: 131_072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (config, job, host)
}
fn quote(config: &Int8SemanticExtractionJobConfig, job: JobLimits, host: JobHostLimits,
    input: JobInputMemory, prefill: Option<Int8PrefillLimits>, threads: Option<usize>)
    -> Result<Int8SemanticJobRequirements, HostedJobError> {
    int8_semantic_job_requirements(config.extraction.budget, &config.options, config.judge_planning,
        &config.encoding, config.encoding_work_limits, config.work, host, job, input, prefill, threads)
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn full_semantic_job_peak_preserves_population_journal_and_both_recipe_buffers() {
    let (config, job, host) = fixture();
    let semantic = semantic_host(host.native, host.preparation_reserve_bytes, config.work)
        .requirements(config.extraction.budget, config.options.specification.judge_budget,
            config.options.limits, config.judge_planning, job.max_input_bytes_per_item).unwrap();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let required = quote(&config, job, host, input, None, None).unwrap();
        assert_eq!(required.native, semantic.native);
        assert_eq!(required.native_workspace_bytes, semantic.native_workspace_bytes);
        assert_eq!(required.preparation_bytes,
            semantic.preparation_bytes + 2 * SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64);
        assert_eq!(required.input_bytes, host.required_buffer_bytes_for_input(job, input).unwrap()
            - host.preparation_reserve_bytes + required.preparation_bytes);
        assert_eq!(required.output_bytes, semantic.output_bytes);
        assert_eq!(required.total_bytes().unwrap(), required.native.kv_bytes
            + required.native_workspace_bytes + required.input_bytes + required.output_bytes);
    }
    let original = quote(&config, job, host, JobInputMemory::CompletePopulation, None, None).unwrap();
    let retained = quote(&config, job, host, JobInputMemory::RetainedRecord, None, None).unwrap();
    assert_eq!(original.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(original.preparation_bytes, retained.preparation_bytes);
    let floor = 2 * SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64 + 1024;
    assert_eq!(preparation_with_recipe(u64::MAX - floor, 1024).unwrap(), u64::MAX);
    assert!(preparation_with_recipe(u64::MAX - floor + 1, 1024).is_err());
}

#[test]
fn all_six_lifetime_work_axes_must_match_before_authentication_or_ingestion() {
    for axis in 0..6 {
        for increase in [false, true] {
            let (config, mut job, host) = fixture();
            let value = match axis {
                0 => &mut job.max_work.model.forward_positions,
                1 => &mut job.max_work.model.projected_logits,
                2 => &mut job.max_work.model.attention_pairs,
                3 => &mut job.max_work.model.projections.dot_products,
                4 => &mut job.max_work.model.projections.multiply_accumulates,
                _ => &mut job.max_work.mask_node_visits,
            };
            if increase { *value += 1; } else { *value -= 1; }
            assert!(quote(&config, job, host, JobInputMemory::CompletePopulation, None, None).is_err(),
                "axis {axis}, increase {increase}");
        }
    }
    let (config, mut job, host) = fixture();
    job.max_result_bytes -= 1;
    assert!(quote(&config, job, host, JobInputMemory::RetainedRecord, None, None).is_err());
}

#[test]
fn compact_document_work_and_invocation_work_add_one_encoder_and_ledger_for_every_schedule() {
    let (mut config, job, host) = fixture();
    let policy = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1_000_000, max_pair_bytes: 4_000_000 };
    let invocation = SafeBpeWorkLimits { max_input_bytes: 65536, max_output_tokens: 32768,
        max_pair_checks: 8_000_000, max_pair_bytes: 32_000_000 };
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        for prompt in [None, Some(prefill(1)), Some(prefill(3))] {
            for threads in [None, Some(2), Some(4)] {
                config.encoding = UntrustedEncoding::ByteFallback; config.encoding_work_limits = None;
                let byte = quote(&config, job, host, input, prompt, threads).unwrap();
                config.encoding = UntrustedEncoding::SafeBpeV1(policy);
                config.encoding_work_limits = Some(invocation);
                let compact = quote(&config, job, host, input, prompt, threads).unwrap();
                assert_eq!(compact.preparation_bytes - byte.preparation_bytes,
                    policy.requirements().unwrap().workspace_bytes + 1024);
                assert_eq!(compact.native, byte.native);
                assert_eq!(compact.native_workspace_bytes, byte.native_workspace_bytes);
                assert_eq!(compact.output_bytes, byte.output_bytes);
                assert_eq!(compact.team_workspace_bytes, byte.team_workspace_bytes);
            }
        }
    }
    config.encoding_work_limits = None;
    assert!(quote(&config, job, host, JobInputMemory::RetainedRecord, None, None).is_err());
}

#[test]
fn immutable_job_compiler_assets_cross_the_same_owned_physical_boundary() {
    fn send<T: Send + 'static>() {}
    send::<SemanticJob>();
    send::<Int8SemanticExtractionJobPlanner>();
    send::<Arc<SemanticAssets>>();
}
