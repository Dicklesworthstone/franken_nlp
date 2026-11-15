//! Model-free memory/admission fixtures. No synthetic identity is ever
//! admitted to a public native model constructor or used for model execution.

use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchLimits, extract::ExtractionMaskBudget,
        sentiment::sampled::{SampledSentimentPlanFacts, sampled_sentiment_batch_preparation_bytes}},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::sentiment::sampled::SampledSentimentLimits,
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding},
};

const PREPARATION: u64 = 600 * 1024 * 1024;
fn corpus() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } }, transport: BatchLimits::default(),
        preparation_reserve_bytes: PREPARATION, io_reserve_bytes: 65_536 }
}
fn task(width: usize) -> TaskBudget {
    TaskBudget { max_input_tokens: 32, max_output_tokens: 2048, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(128, width).unwrap().kv_bytes }
}
fn work(width: usize, samples: usize) -> Int4Work {
    let one = constrained_int4::planned_work(width, 32, 16).unwrap();
    (0..samples).try_fold(Int4Work::default(), |sum, _| sum.checked_add(one)).unwrap()
}
fn work_limits(width: usize, samples: usize) -> SampledSentimentBatchLimits {
    let item = work(width, samples);
    SampledSentimentBatchLimits { max_item_work: item, max_model_work: item.checked_add(item).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 2_000_000,
            checkpoint_interval_nodes: 256 }, max_visits_per_item: 32_000_000_000,
            max_visits_per_run: 64_000_000_000 }, max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn identity(width: usize) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"sampled-host-admission-fixture-only");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width)).unwrap()),
        tokenizer_digest: digest, template_digest: digest, task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "fixture".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: constrained_int4::JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn request(id: &ExecutionIdentity, width: usize) -> SampledSentimentAdmission<'_> {
    SampledSentimentAdmission { facts: SampledSentimentPlanFacts {
        identity: id, model_work: work(width, 8), group_width: width, sample_count: 8,
        required_context: 47, task_budget: task(width), max_result_bytes: 65_536, preparation_bytes: PREPARATION },
        kv_reservation_bytes: task(width).max_kv_bytes, mask_node_visits: 100_000 }
}
fn check(request: &SampledSentimentAdmission<'_>) -> Result<(), BatchItemFailure> {
    check_request(32, 128, task(32).max_kv_bytes, work_limits(32, 8), PREPARATION, 65_536, request)
}

#[test]
fn complete_populations_reuse_one_native_context_and_sampler_across_every_sample() {
    for width in [1, 3, 32, 127, 128] {
        let mut previous = None;
        for samples in [1, 4, 32, 128] {
            let limits = corpus();
            let required = requirements(width, work_limits(width, samples), task(width), PREPARATION, limits).unwrap();
            assert_eq!(required.native, Int4MemoryRequirement::for_context(128, width).unwrap());
            assert_eq!(required.workspace_bytes, required.native.rope_bytes + required.native.scratch_payload_bound
                + limits.native.allocator_reserve_bytes + json_sampling_workspace_bytes().unwrap());
            assert_eq!(required.input_bytes, limits.reservation_bytes().unwrap());
            if let Some(expected) = previous { assert_eq!(required, expected); }
            previous = Some(required);
        }
    }
}

#[test]
fn compact_preparation_quotes_one_encoder_workspace_alongside_largest_axis_bundle() {
    let task = task(32); let limits = SampledSentimentLimits::default();
    let byte = sampled_sentiment_batch_preparation_bytes(task, limits, &UntrustedEncoding::ByteFallback).unwrap();
    let policy = SafeBpeLimits { max_input_bytes: 1024, max_output_tokens: 512,
        max_pair_checks: 65_536, max_pair_bytes: 1 << 20 };
    let compact = sampled_sentiment_batch_preparation_bytes(task, limits,
        &UntrustedEncoding::SafeBpeV1(policy)).unwrap();
    assert_eq!(compact - byte, policy.requirements().unwrap().workspace_bytes);
    let native = work_limits(32, 8);
    let exact = CorpusLimits { preparation_reserve_bytes: compact, ..corpus() };
    requirements(32, native, task, compact, exact).unwrap();
    assert!(requirements(32, native, task, compact,
        CorpusLimits { preparation_reserve_bytes: compact - 1, ..exact }).is_err());
}

#[test]
fn every_resource_axis_refuses_before_native_or_population_access() {
    for axis in 0..5 {
        let mut work = work_limits(32, 8);
        match axis { 0 => work.max_model_work.forward_positions = work.max_item_work.forward_positions - 1,
            1 => work.max_model_work.projected_logits = work.max_item_work.projected_logits - 1,
            2 => work.max_model_work.attention_pairs = work.max_item_work.attention_pairs - 1,
            3 => work.max_model_work.projections.dot_products = work.max_item_work.projections.dot_products - 1,
            _ => work.max_model_work.projections.multiply_accumulates = work.max_item_work.projections.multiply_accumulates - 1 }
        assert!(requirements(32, work, task(32), PREPARATION, corpus()).is_err());
    }
    for axis in 0..9 {
        let mut limits = corpus(); let mut native = work_limits(32, 8); let mut task = task(32);
        match axis { 0 => limits.preparation_reserve_bytes -= 1, 1 => limits.native.context_tokens = 0,
            2 => task.max_kv_bytes -= 1, 3 => native.max_sampler_bytes -= 1,
            4 => native.masks.max_visits_per_run = native.masks.max_visits_per_item - 1,
            5 => limits.io_reserve_bytes = 0, 6 => limits.native.allocator_reserve_bytes = u64::MAX,
            7 => limits.transport.max_json_depth = 65, _ => limits.native.run.max_elapsed = Duration::ZERO }
        assert!(requirements(32, native, task, PREPARATION, limits).is_err(), "axis {axis}");
    }
}

#[test]
fn complete_sample_work_may_exceed_context_but_every_admission_binding_is_checked() {
    let id = identity(32); let admitted = request(&id, 32);
    assert!(admitted.facts.model_work.forward_positions > 128);
    check(&admitted).unwrap();
    for axis in 0..9 {
        let mut request = request(&id, 32);
        match axis { 0 => request.facts.required_context = 129,
            1 => request.kv_reservation_bytes -= 1, 2 => request.facts.task_budget.max_kv_bytes -= 1,
            3 => request.facts.preparation_bytes += 1, 4 => request.mask_node_visits = 0,
            5 => request.mask_node_visits = work_limits(32, 8).masks.max_visits_per_item + 1,
            6 => request.facts.sample_count = 0, 7 => request.facts.model_work.projections.dot_products += 1,
            _ => request.facts.group_width = 128 }
        assert!(check(&request).unwrap_err().stop, "axis {axis}");
    }
    let mut wrong = identity(32); wrong.sampler_version = "distribution-scoring".to_owned();
    assert!(check(&request(&wrong, 32)).unwrap_err().stop);
    let mut request = request(&id, 32); request.facts.max_result_bytes += 1;
    assert_eq!(check(&request).unwrap_err().fault.code, BatchCode::OutputLineLimit);
}

#[test]
fn owned_planner_and_io_cross_one_physical_serial_or_team_boundary() {
    fn send<T: Send + 'static>() {}
    send::<SampledSentimentBatchPlanner>();
    send::<Arc<SampledSentimentVocabulary>>();
    send::<SampledSentimentRun>();
    let _serial = NlpEngine::batch_int4_sampled_sentiment::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int4_sampled_sentiment_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _prefill = NlpEngine::batch_int4_sampled_sentiment_layer_major::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    for width in [0, 1, 65, usize::MAX] { assert!(int4_task_team_workspace_bytes(width, 32).is_err()); }
    for width in [2, 4, 64] {
        let worker = int4_task_team_workspace_bytes(width, 32).unwrap();
        let required = requirements(32, work_limits(32, 8), task(32), PREPARATION, corpus()).unwrap();
        assert!(sum(&[required.workspace_bytes, worker]).unwrap() > required.workspace_bytes);
    }
}

#[test]
fn corpus_prefill_prices_the_full_declared_morsel_once_across_all_records_and_samples() {
    for width in [1, 3, 32, 128] {
        let base = requirements(width, work_limits(width, 1), task(width), PREPARATION, corpus()).unwrap();
        for rows in [1, 4, 64] {
            let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            for samples in [1, 8, 128] {
                let required = scheduled_requirements(width, work_limits(width, samples),
                    task(width), PREPARATION, corpus(), Some(prefill)).unwrap();
                // The task prompt ceiling is 32, so a 64-row declaration also
                // checks that short prompts cannot shrink the admission quote.
                assert_eq!(required.native, base.native);
                assert_eq!(required.input_bytes, base.input_bytes);
                assert_eq!(required.workspace_bytes, base.workspace_bytes + extra);
            }
        }
        assert_eq!(scheduled_requirements(width, work_limits(width, 1), task(width),
            PREPARATION, corpus(), None).unwrap(), base);
    }
}

#[test]
fn corpus_prefill_refuses_invalid_rows_and_short_workspace_before_population_allocation() {
    for rows in [0, 65, usize::MAX] {
        let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
        assert!(scheduled_requirements(32, work_limits(32, 8), task(32),
            PREPARATION, corpus(), Some(prefill)).is_err());
    }
    let extra = Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap();
    assert!(scheduled_requirements(32, work_limits(32, 8), task(32), PREPARATION,
        corpus(), Some(Int4PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: extra - 1 })).is_err());
    let mut short_kv = task(32);
    short_kv.max_kv_bytes -= 1;
    assert!(scheduled_requirements(32, work_limits(32, 8), short_kv, PREPARATION,
        corpus(), Some(Int4PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: extra })).is_err());
}
